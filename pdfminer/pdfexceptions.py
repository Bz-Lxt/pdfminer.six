from pdfminer.psexceptions import PSException


class PDFException(PSException):
    pass


class PDFTypeError(PDFException, TypeError):
    pass


class PDFValueError(PDFException, ValueError):
    pass


class PDFObjectNotFound(PDFException):
    pass


class PDFNotImplementedError(PDFException, NotImplementedError):
    pass


class PDFKeyError(PDFException, KeyError):
    pass


class PDFEOFError(PDFException, EOFError):
    pass


class PDFIOError(PDFException, IOError):
    pass


class PDFUnavailableRangeError(PDFException):
    """A byte range of the PDF is currently unavailable.

    Raised by byte-range sources backing a partially downloaded linearized
    PDF when a requested offset range cannot be served (for example, because
    the download has not progressed that far yet and the source does not
    block until it does).
    """
