'''
Raise error without traceback
'''
import sys

class QuietError(Exception):
    # All who inherit me shall not traceback, but be spoken of cleanly
    pass

class ParseError(QuietError):
    # Failed to parse data
    pass

class ArgumentError(QuietError):
    # Some other problem with arguments
    pass

class WriteError(QuietError):
    # Access violation
    pass

def quiet_hook(kind, message, traceback):
    if QuietError in kind.__bases__:
        print('{0}: {1}'.format(kind.__name__, message))  # Only print Error Type and Message
    else:
        sys.__excepthook__(kind, message, traceback)  # Print Error Type, Message and Traceback

sys.excepthook = quiet_hook
