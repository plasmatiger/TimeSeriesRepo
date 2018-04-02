from numpy import arange, array, ones
from numpy.linalg import lstsq

def leastsquareslinefit(sequence,seq_range):
    y = array(sequence[seq_range[0]:seq_range[1]+1])
    A = ones((len(x),2),float)
    A[:,0] = x
    (p,residuals,rank,s) = lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
return (p,error)


def sumsquared_error(sequence, segment):
    x0,y0,x1,y1 = segment
    p, error = leastsquareslinefit(sequence,(x0,x1))
    return error


def regression(sequence, seq_range):
    p, error = leastsquareslinefit(sequence,seq_range)
    y0 = p[0]*seq_range[0] + p[1]
    y1 = p[0]*seq_range[1] + p[1]
    return (seq_range[0],y0,seq_range[1],y1)


def slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
    if not seq_range:
        seq_range = (0,len(sequence)-1)

    start = seq_range[0]
    end = start
    result_segment = create_segment(sequence,(seq_range[0],seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment = create_segment(sequence,(start,end))
        error = compute_error(sequence,test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
return [result_segment] + slidingwindowsegment(sequence, create_segment, compute_error, max_error, (end-1,seq_range[1]))


x, y = datasets.data_reader.read_clean_dataset(summary=True)

#xp = np.ndarray((30000,), dtype = 'float64')

xp = []
max_error = 0.005
for i in x:
    segment = slidingwindowsegment(i, regression, sumsquared_error, max_error)
    xp.append(segment)