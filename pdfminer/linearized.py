"""Streaming page extraction from linearized ("Fast Web View") PDFs.

A linearized PDF (PDF 1.4, appendix F) starts with a linearization
parameter dictionary and a cross-reference section that only describes
the objects needed to render the first page.  The first page can
therefore be parsed as soon as the beginning of the file has been
downloaded; every other page section follows in document order.  The
hint stream embedded in the first-page section describes the byte range
of each page and of the shared object groups.

This module implements that streaming read on top of a
:class:`ByteRangeSource`, an object that can return arbitrary byte
ranges of the (partially downloaded) file.  Pages are yielded in
document order, the first page first, without gaps or duplicates even
when the underlying blocks arrive out of order.  If the file turns out
not to be linearized or the hint tables are corrupt, the complete file
is read through the source and parsing falls back to the regular code
path, yielding exactly the same results as a local file.
"""

import io
import logging
from collections.abc import Container, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTPage
from pdfminer.pdfdocument import PDFDocument, PDFNoValidXRef
from pdfminer.pdfexceptions import (
    PDFException,
    PDFUnavailableRangeError,
)
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdftypes import PDFStream, int_value, list_value
from pdfminer.psexceptions import PSEOF
from pdfminer.psparser import KWD, PSKeyword

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

log = logging.getLogger(__name__)

# The linearization parameter dictionary has to appear within the first
# 1024 bytes of the file (PDF spec, appendix F.2).
LINEARIZATION_PROBE_SIZE = 1024
# Chunk size used when reading the complete file for the fallback path.
FULL_READ_CHUNK_SIZE = 64 * 1024


class _LinearizationUnusable(PDFException):
    """Internal signal: streaming is not possible, read the whole file."""


@runtime_checkable
class ByteRangeSource(Protocol):
    """A source that serves arbitrary byte ranges of a PDF.

    Implementations return ``length`` bytes starting at ``offset``; the
    returned ``bytes`` object must begin at ``offset`` and is allowed to
    contain more bytes than requested (the extras are cached).  Calls may
    block until the requested range has been downloaded.  Byte ranges may
    be requested in any order and the same range may be requested more
    than once; overlapping or repeated deliveries must always describe the
    same bytes.  A short (empty) read marks permanent EOF.  If a range is
    not available yet and the implementation cannot block, it should raise
    :class:`~pdfminer.pdfexceptions.PDFUnavailableRangeError`.
    """

    def read_range(self, offset: int, length: int) -> bytes: ...


class SparseRangeIO(io.RawIOBase):
    """Binary IO over a byte-range source with sparse, out-of-order data.

    Downloaded byte intervals are cached in an ordered, disjoint interval
    map.  Blocks may be added in any order, repeatedly, or with overlaps;
    adjacent blocks are merged.  Reads serve contiguous cached bytes and,
    when a gap is reached, pull the missing range from the underlying
    source.  The object is seekable and can be used wherever pdfminer
    expects a binary file object.
    """

    def __init__(self, source: ByteRangeSource, size: int | None = None) -> None:
        super().__init__()
        self._source = source
        self._size = size
        self._blocks: dict[int, bytes] = {}
        self._pos = 0
        # Soft high-water mark for parser-driven reads.  Bytes past the
        # mark are only served from the cache (blocks delivered out of
        # order or explicitly prefetched); missing bytes are never pulled
        # from the source past it.  This keeps the tokenizer's readahead
        # from downloading page sections that have not been requested yet.
        self._read_limit = size

    def set_read_limit(self, offset: int) -> None:
        """Allow parser-driven reads up to *offset* bytes from the start."""
        self._read_limit = offset

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            if self._size is None:
                msg = "cannot seek from end without a declared file size"
                raise PDFUnavailableRangeError(msg)
            pos = self._size + offset
        else:
            msg = f"invalid whence value: {whence!r}"
            raise ValueError(msg)
        if pos < 0:
            msg = f"negative seek position: {pos}"
            raise ValueError(msg)
        self._pos = pos
        return pos

    def add_range(self, offset: int, data: bytes) -> None:
        """Add a downloaded block, merging it with previously cached data."""
        if not data:
            return
        if offset < 0:
            msg = f"negative block offset: {offset}"
            raise ValueError(msg)
        end = offset + len(data)
        # A repeated delivery of bytes that are already cached is a no-op.
        cached = self._find_block(offset)
        if cached is not None:
            start, block = cached
            if start + len(block) >= end:
                return
        # Drop overlaps with existing blocks, preserving non-overlapping
        # tails.  Blocks are assumed to describe identical bytes.
        for start in tuple(self._blocks):
            block = self._blocks[start]
            stop = start + len(block)
            if stop <= offset or start >= end:
                continue
            del self._blocks[start]
            if start < offset:
                self._blocks[start] = block[: offset - start]
            if stop > end:
                self._blocks[end] = block[end - start :]
        # Merge with an adjacent block on the left.
        merged = bytearray(data)
        new_offset = offset
        for start, block in tuple(self._blocks.items()):
            if start + len(block) == new_offset:
                merged[:0] = block
                new_offset = start
                del self._blocks[start]
        # Merge with an adjacent block on the right.
        right = new_offset + len(merged)
        if right in self._blocks:
            merged.extend(self._blocks.pop(right))
        self._blocks[new_offset] = bytes(merged)

    def available(self, offset: int) -> int:
        """Return how many contiguous bytes are cached at *offset*."""
        block = self._find_block(offset)
        if block is None:
            return 0
        start, data = block
        return len(data) - (offset - start)

    def _find_block(self, pos: int) -> tuple[int, bytes] | None:
        for start, data in self._blocks.items():
            if start <= pos < start + len(data):
                return (start, data)
        return None

    def _fetch(self, pos: int, size: int) -> bytes:
        """Pull a range straight from the source, ignoring the read limit."""
        if self._size is not None:
            size = min(size, self._size - pos)
        if size <= 0:
            return b""
        data = self._source.read_range(pos, size)
        if data:
            self.add_range(pos, data)
        return data

    def _serve(self, pos: int, size: int) -> bytes:
        result = bytearray()
        cur = pos
        while len(result) < size:
            if self._size is not None and cur >= self._size:
                break
            block = self._find_block(cur)
            if block is not None:
                # Cached data is always readable, even beyond the read
                # limit: explicit prefetches and out-of-order deliveries.
                start, data = block
                take = min(len(data) - (cur - start), size - len(result))
                result.extend(data[cur - start : cur - start + take])
                cur += take
                continue
            if self._read_limit is not None and cur >= self._read_limit:
                # The parser is trying to read a section that has not been
                # released yet; present an EOF boundary instead of pulling
                # bytes for later pages.
                break
            wanted = size - len(result)
            if self._read_limit is not None:
                wanted = min(wanted, self._read_limit - cur)
            data = self._fetch(cur, wanted)
            if not data:
                # Permanent EOF.
                break
        return bytes(result)

    def readinto(self, b: WriteableBuffer) -> int:
        view = memoryview(b).cast("B")
        data = self._serve(self._pos, len(view))
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            if self._size is None:
                # Read until permanent EOF.
                result = bytearray()
                while True:
                    chunk = self._serve(self._pos, FULL_READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    result.extend(chunk)
                    self._pos += len(chunk)
                return bytes(result)
            size = self._size - self._pos
        data = self._serve(self._pos, size)
        self._pos += len(data)
        return data

    def ensure_available(self, start: int, end: int) -> None:
        """Make sure the contiguous range [start, end) can be read.

        Missing sub-ranges are fetched directly from the source, ignoring
        the parser read limit.  Ranges already cached (including blocks
        delivered out of order) are not requested again.
        """
        pos = start
        while pos < end:
            available = self.available(pos)
            if available:
                pos += min(available, end - pos)
                continue
            gap = min(end - pos, FULL_READ_CHUNK_SIZE)
            data = self._fetch(pos, gap)
            if not data:
                msg = f"byte range [{pos}, {pos + gap}) is not available"
                raise PDFUnavailableRangeError(msg)
            pos += len(data)


@dataclass
class _SharedGroup:
    """One entry of the shared object hint table."""

    objid: int
    start: int
    length: int
    nobjects: int


@dataclass
class _PageHint:
    """Byte footprint of one page as described by the hint stream."""

    nobjects: int
    start: int
    length: int
    shared_groups: list[int] = field(default_factory=list)


@dataclass
class _LinearizationInfo:
    """Parsed linearization dictionary and hint tables."""

    length: int
    first_page_obj: int
    first_page_end: int
    pages_total: int
    hint_offset: int
    hint_length: int
    pages: list[_PageHint]
    groups: list[_SharedGroup]


class _BitReader:
    """Big-endian, most-significant-bit-first bit reader for hint tables."""

    __slots__ = ("data", "bitpos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bitpos = 0

    def bits(self, nbits: int) -> int:
        value = 0
        for _ in range(nbits):
            bytepos = self.bitpos >> 3
            if bytepos >= len(self.data):
                msg = "hint table overruns its stream"
                raise _LinearizationUnusable(msg)
            value = (value << 1) | (
                (self.data[bytepos] >> (7 - (self.bitpos & 7))) & 1
            )
            self.bitpos += 1
        return value

    def align(self) -> None:
        if self.bitpos & 7:
            self.bitpos = (self.bitpos + 7) & ~7


def _read_vector(reader: _BitReader, count: int, nbits: int) -> list[int]:
    return [reader.bits(nbits) for _ in range(count)]


def _decode_page_offset_table(
    reader: _BitReader,
    npages: int,
) -> tuple[int, int, int, list[_PageHint]]:
    min_nobjects = reader.bits(32)
    first_page_offset = reader.bits(32)
    nbits_delta_nobjects = reader.bits(16)
    min_page_length = reader.bits(32)
    nbits_delta_page_length = reader.bits(16)
    reader.bits(32)  # min_content_offset (implementation note 126: wrong)
    nbits_delta_content_offset = reader.bits(16)
    reader.bits(32)  # min_content_length
    nbits_delta_content_length = reader.bits(16)
    nbits_nshared_objects = reader.bits(16)
    nbits_shared_identifier = reader.bits(16)
    nbits_shared_numerator = reader.bits(16)
    reader.bits(16)  # shared_denominator (unused)
    for nbits in (
        nbits_delta_nobjects,
        nbits_delta_page_length,
        nbits_delta_content_offset,
        nbits_delta_content_length,
        nbits_nshared_objects,
        nbits_shared_identifier,
        nbits_shared_numerator,
    ):
        if nbits > 32:
            msg = "page offset hint table uses more than 32 bits per field"
            raise _LinearizationUnusable(msg)

    delta_nobjects = _read_vector(reader, npages, nbits_delta_nobjects)
    reader.align()
    delta_page_length = _read_vector(reader, npages, nbits_delta_page_length)
    reader.align()
    nshared_objects = _read_vector(reader, npages, nbits_nshared_objects)
    reader.align()
    shared_identifiers: list[list[int]] = []
    for i in range(npages):
        ids = _read_vector(
            reader, nshared_objects[i], nbits_shared_identifier
        )
        shared_identifiers.append(ids)
    reader.align()
    for i in range(npages):
        # shared numerators are read but not used (qpdf always writes 0)
        _read_vector(reader, nshared_objects[i], nbits_shared_numerator)
    reader.align()
    _read_vector(reader, npages, nbits_delta_content_offset)
    reader.align()
    _read_vector(reader, npages, nbits_delta_content_length)
    reader.align()

    pages = [
        _PageHint(
            nobjects=min_nobjects + delta_nobjects[i],
            start=0,
            length=min_page_length + delta_page_length[i],
            shared_groups=shared_identifiers[i],
        )
        for i in range(npages)
    ]
    return min_page_length, first_page_offset, min_nobjects, pages


def _decode_shared_object_table(
    reader: _BitReader,
    first_page_obj: int,
    first_page_offset: int,
    hint_offset: int,
    hint_length: int,
) -> tuple[int, int, list[_SharedGroup]]:
    first_shared_obj = reader.bits(32)
    first_shared_offset = reader.bits(32)
    nshared_first_page = reader.bits(32)
    nshared_total = reader.bits(32)
    nbits_nobjects = reader.bits(16)
    min_group_length = reader.bits(32)
    nbits_delta_group_length = reader.bits(16)
    if nbits_nobjects > 32 or nbits_delta_group_length > 32:
        msg = "shared object hint table uses more than 32 bits per field"
        raise _LinearizationUnusable(msg)
    if nshared_total < nshared_first_page or nshared_total < 0:
        msg = "inconsistent shared object counts in hint table"
        raise _LinearizationUnusable(msg)

    delta_group_length = _read_vector(
        reader, nshared_total, nbits_delta_group_length
    )
    reader.align()
    signature_present = _read_vector(reader, nshared_total, 1)
    for present in signature_present:
        if present:
            # An optional MD5 signature of the group.
            reader.bits(128)
    reader.align()
    nobjects_minus_one = _read_vector(reader, nshared_total, nbits_nobjects)
    reader.align()

    def adjust(offset: int) -> int:
        # Hint offsets disregard the hint stream itself.
        if offset >= hint_offset:
            return offset + hint_length
        return offset

    groups: list[_SharedGroup] = []
    cur_obj = first_page_obj
    cur_pos = adjust(first_page_offset)
    for i in range(nshared_total):
        if i == nshared_first_page:
            cur_obj = first_shared_obj
            cur_pos = adjust(first_shared_offset)
        length = min_group_length + delta_group_length[i]
        nobjects = nobjects_minus_one[i] + 1
        groups.append(
            _SharedGroup(
                objid=cur_obj,
                start=cur_pos,
                length=length,
                nobjects=nobjects,
            )
        )
        cur_obj += nobjects
        cur_pos += length
    return nshared_first_page, nshared_total, groups


def _parse_linearization_dict(probe: bytes) -> dict[str, object]:
    """Parse the first indirect object, requiring a linearization dict."""
    try:
        parser = PDFParser(io.BytesIO(probe))
        (_, objid) = parser.nexttoken()
        (_, _genno) = parser.nexttoken()
        (_, keyword) = parser.nexttoken()
        if not isinstance(objid, int) or not isinstance(keyword, PSKeyword):
            msg = "linearization parameter dictionary not found"
            raise _LinearizationUnusable(msg)
        if keyword is not KWD(b"obj"):
            msg = "linearization parameter dictionary not found"
            raise _LinearizationUnusable(msg)
        (_, obj) = parser.nextobject()
    except (PSEOF, PDFSyntaxError) as err:
        msg = "linearization parameter dictionary not found"
        raise _LinearizationUnusable(msg) from err
    if not isinstance(obj, dict) or int_value(obj.get("Linearized", 0)) != 1:
        msg = "file is not linearized"
        raise _LinearizationUnusable(msg)
    for key in ("L", "H", "O", "E", "N", "T"):
        if key not in obj:
            msg = f"linearization dictionary is missing /{key}"
            raise _LinearizationUnusable(msg)
    return obj


def _lint_params(lin: dict[str, object]) -> tuple[int, list[int], int, int, int]:
    length = int_value(lin["L"])
    hint_pairs = list_value(lin["H"])
    first_page_obj = int_value(lin["O"])
    first_page_end = int_value(lin["E"])
    pages_total = int_value(lin["N"])
    if length <= 0 or first_page_obj < 0 or first_page_end <= 0:
        msg = "invalid linearization parameters"
        raise _LinearizationUnusable(msg)
    if pages_total < 1:
        msg = "linearization dictionary declares no pages"
        raise _LinearizationUnusable(msg)
    if len(hint_pairs) not in (2, 4):
        msg = "/H must be an array of two or four integers"
        raise _LinearizationUnusable(msg)
    hints = [int_value(value) for value in hint_pairs]
    for i in range(0, len(hints), 2):
        if hints[i] < 0 or hints[i + 1] <= 0:
            msg = "invalid hint stream location in linearization dictionary"
            raise _LinearizationUnusable(msg)
        if hints[i] + hints[i + 1] > length:
            msg = "hint stream extends past the declared file length"
            raise _LinearizationUnusable(msg)
    return length, hints, first_page_obj, first_page_end, pages_total


def _locate_first_xref(parser: PDFParser, first_page_end: int) -> int:
    """Find the first-page cross-reference section right after the lindict."""
    # Skip over the linearization parameter dictionary itself.
    try:
        (_, objid) = parser.nexttoken()
        (_, _genno) = parser.nexttoken()
        (_, keyword) = parser.nexttoken()
        if not isinstance(objid, int) or keyword is not KWD(b"obj"):
            msg = "linearization parameter dictionary not found"
            raise _LinearizationUnusable(msg)
        parser.nextobject()
    except (PSEOF, PDFSyntaxError) as err:
        msg = "linearization parameter dictionary not found"
        raise _LinearizationUnusable(msg) from err
    while True:
        try:
            (pos, token) = parser.nexttoken()
        except PSEOF as err:
            msg = "first-page cross-reference section not found"
            raise _LinearizationUnusable(msg) from err
        if pos >= first_page_end:
            msg = "first-page cross-reference section starts past /E"
            raise _LinearizationUnusable(msg)
        if isinstance(token, int):
            # A cross-reference stream ("N G obj ...") starts here.
            return pos
        if isinstance(token, PSKeyword) and token is KWD(b"xref"):
            return pos


def _read_hint_stream(
    parser: PDFParser,
    hints: list[int],
) -> bytes:
    raw = bytearray()
    for i in range(0, len(hints), 2):
        offset = hints[i]
        parser.seek(offset)
        try:
            parser.nexttoken()  # object id
            parser.nexttoken()  # generation number
            (_, keyword) = parser.nexttoken()
            if keyword is not KWD(b"obj"):
                msg = "/H does not point at an object"
                raise _LinearizationUnusable(msg)
            (_, obj) = parser.nextobject()
        except (PSEOF, PDFSyntaxError) as err:
            msg = "hint stream cannot be parsed"
            raise _LinearizationUnusable(msg) from err
        if not isinstance(obj, PDFStream):
            msg = "/H does not point at a stream object"
            raise _LinearizationUnusable(msg)
        try:
            data = obj.get_data()
        except PDFException as err:
            msg = "hint stream cannot be decoded"
            raise _LinearizationUnusable(msg) from err
        # The stream length may differ from the declared /H length; use the
        # decoded data.  Both hint halves (primary and overflow) are part of
        # one logical bit stream.
        raw.extend(data)
    return bytes(raw)


def _build_info(
    length: int,
    hints: list[int],
    first_page_obj: int,
    first_page_end: int,
    pages_total: int,
    doc: PDFDocument,
    hint_data: bytes,
) -> _LinearizationInfo:
    hint_offset = hints[0]
    hint_length = hints[1] + (hints[3] if len(hints) == 4 else 0)

    try:
        reader = _BitReader(hint_data)
        (
            _min_page_length,
            first_page_offset,
            _min_nobjects,
            pages,
        ) = _decode_page_offset_table(reader, pages_total)
        (
            nshared_first_page,
            nshared_total,
            groups,
        ) = _decode_shared_object_table(
            reader,
            first_page_obj,
            first_page_offset,
            hint_offset,
            hint_length,
        )
    except _LinearizationUnusable:
        raise
    except (IndexError, ValueError, OverflowError) as err:
        msg = "hint table is corrupt"
        raise _LinearizationUnusable(msg) from err

    def adjust(offset: int) -> int:
        if offset >= hint_offset:
            return offset + hint_length
        return offset

    page0_start = adjust(first_page_offset)
    # The hint table offsets disregard the hint stream, so the page
    # offsets have to be shifted as well.  Part 8 (shared objects used by
    # the remaining pages) sits between the first page section and the
    # second page section.
    part8_length = sum(
        group.length
        for group in groups[nshared_first_page:nshared_total]
    )
    pages[0].start = page0_start
    second_page_start = page0_start + pages[0].length + part8_length
    for i in range(1, pages_total):
        pages[i].start = second_page_start + sum(
            pages[j].length for j in range(1, i)
        )

    # Validate against the authoritative first-page cross-reference.
    try:
        (strmid, pos, _genno) = doc.xrefs[0].get_pos(first_page_obj)
    except KeyError as err:
        msg = "first page object missing from first-page xref"
        raise _LinearizationUnusable(msg) from err
    if strmid is not None or pos != page0_start:
        msg = "first page offset in hint table does not match the xref"
        raise _LinearizationUnusable(msg)
    if pages[0].start + pages[0].length != first_page_end:
        msg = "first page length in hint table does not match /E"
        raise _LinearizationUnusable(msg)
    for page in pages:
        if page.start < 0 or page.length <= 0:
            msg = "invalid page range in hint table"
            raise _LinearizationUnusable(msg)
        if page.start + page.length > length:
            msg = "page range extends past declared file length"
            raise _LinearizationUnusable(msg)
        for group_id in page.shared_groups:
            if not 0 <= group_id < nshared_total:
                msg = "page references unknown shared object group"
                raise _LinearizationUnusable(msg)
    for group in groups:
        if group.start < 0 or group.length <= 0:
            msg = "invalid shared object group range in hint table"
            raise _LinearizationUnusable(msg)
        if group.start + group.length > length:
            msg = "shared object group extends past declared file length"
            raise _LinearizationUnusable(msg)

    return _LinearizationInfo(
        length=length,
        first_page_obj=first_page_obj,
        first_page_end=first_page_end,
        pages_total=pages_total,
        hint_offset=hint_offset,
        hint_length=hint_length,
        pages=pages,
        groups=groups,
    )


def _read_full_file(
    source: ByteRangeSource,
    length: int | None,
) -> bytes:
    result = bytearray()
    pos = 0
    while length is None or pos < length:
        wanted = FULL_READ_CHUNK_SIZE
        if length is not None:
            wanted = min(wanted, length - pos)
        chunk = source.read_range(pos, wanted)
        if not chunk:
            break
        result.extend(chunk)
        pos += len(chunk)
    if length is not None and len(result) > length:
        return bytes(result[:length])
    return bytes(result)


def _is_wanted(index: int, page_numbers: Container[int] | None) -> bool:
    return page_numbers is None or index in page_numbers


def _stream_pages(
    source: ByteRangeSource,
    password: str,
    page_numbers: Container[int] | None,
    caching: bool,
    laparams: LAParams,
) -> Iterator[tuple[object, LTPage]]:
    """Yield (pageid, LTPage) using linearized streaming; raise on failure."""
    resource_manager = PDFResourceManager(caching=caching)
    device = PDFPageAggregator(resource_manager, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)

    probe = source.read_range(0, LINEARIZATION_PROBE_SIZE)
    if not probe:
        msg = "source returned no data"
        raise _LinearizationUnusable(msg)
    lin = _parse_linearization_dict(probe)
    (
        length,
        hints,
        first_page_obj,
        first_page_end,
        pages_total,
    ) = _lint_params(lin)

    sparse = SparseRangeIO(source, length)
    sparse.add_range(0, probe)
    # Until the first page has been yielded, parser-driven reads must not
    # run past the end of the first-page section (/E): this keeps the
    # tokenizer's readahead from pulling bytes of later pages.
    sparse.set_read_limit(first_page_end)
    # Fetch the whole first-page section (parts 1 to 6).
    sparse.ensure_available(len(probe), first_page_end)

    parser = PDFParser(sparse)
    xref_pos = _locate_first_xref(parser, first_page_end)
    try:
        doc = PDFDocument(
            parser,
            password=password,
            caching=caching,
            fallback=False,
            xref_pos=xref_pos,
            follow_prev=False,
        )
    except (PSEOF, PDFSyntaxError) as err:
        # Password errors are not caught here: the full file would fail
        # the same way, so there is no point in falling back.
        msg = "first-page cross-reference section is unusable"
        raise _LinearizationUnusable(msg) from err
    if not doc.xrefs:
        msg = "first-page cross-reference section is unusable"
        raise _LinearizationUnusable(msg)

    hint_data = _read_hint_stream(parser, hints)
    info = _build_info(
        length,
        hints,
        first_page_obj,
        first_page_end,
        pages_total,
        doc,
        hint_data,
    )

    # Render and yield the first page directly; the shared /Pages tree
    # lives in the not-yet-downloaded part 9 and is not needed for it.
    try:
        page0_attrs = doc.getobj(first_page_obj)
    except (PSEOF, PDFException) as err:
        msg = "first page object cannot be read"
        raise _LinearizationUnusable(msg) from err
    if not isinstance(page0_attrs, dict):
        msg = "first page object is not a page dictionary"
        raise _LinearizationUnusable(msg)
    if _is_wanted(0, page_numbers):
        page0 = PDFPage(doc, first_page_obj, page0_attrs, None)
        try:
            interpreter.process_page(page0)
        except (PSEOF, PDFException) as err:
            msg = "first page cannot be processed"
            raise _LinearizationUnusable(msg) from err
        yield (first_page_obj, device.get_result())

    if pages_total == 1:
        return

    # Largest document page index that still has to be produced.  When a
    # page selection is given there is no need to download anything past
    # the last selected page.
    last_index = pages_total - 1
    if page_numbers is not None:
        while last_index >= 1 and last_index not in page_numbers:
            last_index -= 1
    if last_index < 1:
        return

    # The tail holds the remaining ("other") objects, the main
    # cross-reference section and the main trailer.  It is fetched before
    # the second page so that the /Pages tree can be walked; the parser
    # read limit stays at /E, these bytes are only served from the cache.
    last_page_end = info.pages[-1].start + info.pages[-1].length
    trailer = doc.xrefs[0].get_trailer()
    try:
        main_xref = int_value(trailer["Prev"])
    except KeyError as err:
        msg = "first-page trailer has no /Prev pointer"
        raise _LinearizationUnusable(msg) from err
    if not last_page_end <= main_xref <= length:
        msg = "/Prev pointer outside the document tail"
        raise _LinearizationUnusable(msg)
    sparse.ensure_available(last_page_end, length)
    try:
        doc.append_xref(parser, main_xref)
    except (PSEOF, PDFSyntaxError) as err:
        msg = "main cross-reference section is unusable"
        raise _LinearizationUnusable(msg) from err

    # Drive the /Pages tree in document order, prefetching each page's
    # byte footprint (and its shared object groups) before it is parsed.
    tree = iter(PDFPage.create_pages(doc))
    try:
        first_tree_page = next(tree)
    except (PSEOF, PDFException) as err:
        msg = "page tree cannot be read"
        raise _LinearizationUnusable(msg) from err
    if first_tree_page.pageid != first_page_obj:
        msg = "page tree order does not match linearized first page"
        raise _LinearizationUnusable(msg)

    seen_ids = {first_page_obj}
    for index in range(1, last_index + 1):
        hint = info.pages[index]
        page_end = hint.start + hint.length
        # Release the page's byte footprint (and its shared groups) to
        # parser-driven reads only after it has been prefetched.  Bytes of
        # later pages stay out of reach until their own turn.
        sparse.ensure_available(hint.start, page_end)
        for group_id in hint.shared_groups:
            group = info.groups[group_id]
            # Groups of the first-page section are already downloaded.
            if group.start + group.length <= first_page_end:
                continue
            sparse.ensure_available(group.start, group.start + group.length)
        sparse.set_read_limit(page_end)
        try:
            page = next(tree)
        except StopIteration as err:
            msg = "page tree has fewer pages than the hint table declares"
            raise _LinearizationUnusable(msg) from err
        except (PSEOF, PDFException) as err:
            msg = f"page {index} cannot be read"
            raise _LinearizationUnusable(msg) from err
        if page.pageid in seen_ids:
            msg = "page tree yields a page more than once"
            raise _LinearizationUnusable(msg)
        seen_ids.add(page.pageid)
        if _is_wanted(index, page_numbers):
            try:
                interpreter.process_page(page)
            except (PSEOF, PDFException) as err:
                msg = f"page {index} cannot be processed"
                raise _LinearizationUnusable(msg) from err
            yield (page.pageid, device.get_result())

    if last_index == pages_total - 1:
        # The whole tree was walked; the page count must agree with /N.
        try:
            extra = next(tree)
        except StopIteration:
            extra = None
        except (PSEOF, PDFException) as err:
            msg = "page tree cannot be read"
            raise _LinearizationUnusable(msg) from err
        if extra is not None:
            msg = "page tree has more pages than the hint table declares"
            raise _LinearizationUnusable(msg)


def extract_pages_range(
    source: ByteRangeSource,
    password: str = "",
    page_numbers: Container[int] | None = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: LAParams | None = None,
) -> Iterator[LTPage]:
    """Extract LTPage objects from a partially downloaded linearized PDF.

    Behaves like
    :func:`pdfminer.high_level.extract_pages`, but takes a
    :class:`ByteRangeSource` instead of a file.  The first page is yielded
    as soon as the first-page section is available; subsequent pages are
    fetched in document order.  If the document is not linearized or its
    hint tables are corrupt, the complete file is downloaded and the
    regular extraction path is used, so results stay identical to reading
    the local file.
    """
    if laparams is None:
        laparams = LAParams()

    yielded_ids: set[object] = set()
    emitted = 0
    stream = _stream_pages(
        source,
        password=password,
        page_numbers=page_numbers,
        caching=caching,
        laparams=laparams,
    )
    while True:
        try:
            page_id, layout = next(stream)
        except _LinearizationUnusable as err:
            log.info(
                "linearized streaming unavailable (%s); "
                "reading the complete file",
                err,
            )
            break
        except StopIteration:
            return
        yielded_ids.add(page_id)
        yield layout
        emitted += 1
        if maxpages and emitted >= maxpages:
            return

    # Fallback: read the whole file and replay the pages that have not
    # been yielded yet.  Object numbers are stable, so page ids match.
    # Start the page counter past the already emitted pages so that the
    # resulting LTPage numbers match a run on the complete local file.
    data = _read_full_file(source, None)
    resource_manager = PDFResourceManager(caching=caching)
    device = PDFPageAggregator(
        resource_manager, pageno=emitted + 1, laparams=laparams
    )
    interpreter = PDFPageInterpreter(resource_manager, device)
    for page in PDFPage.get_pages(
        io.BytesIO(data),
        page_numbers,
        maxpages=0,
        password=password,
        caching=caching,
    ):
        if page.pageid in yielded_ids:
            continue
        if maxpages and emitted >= maxpages:
            break
        interpreter.process_page(page)
        yielded_ids.add(page.pageid)
        emitted += 1
        yield device.get_result()
