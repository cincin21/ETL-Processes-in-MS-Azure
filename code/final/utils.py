import time

def str_time_prop(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    start_time = time.mktime(time.strptime(start, format))
    end_time = time.mktime(time.strptime(end, format))

    ptime = start_time + prop * (end_time - start_time)

    return time.strftime(format, time.localtime(ptime))


def random_date(start, end, prop):
    # return str_time_prop(start, end, '%m/%d/%Y %I:%M %p', prop)
    return str_time_prop(start, end, "%Y%m%d_%H%M%S", prop)