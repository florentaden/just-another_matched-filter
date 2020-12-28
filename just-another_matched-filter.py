import numpy as np
import pandas as pd
import matplotlib.dates as md

from mpi4py import MPI
from sys import argv
from obspy import read
from time import time
from glob import glob
from statsmodels.robust import mad

def get_norm(template, data):
    Ntmp = len(template)
    norm_tmp = np.sum(template**2)
    data_series = pd.Series(data**2)
    norm_data = data_series.rolling(Ntmp).sum()[Ntmp-1:]
    return np.sqrt(norm_data*norm_tmp)

def xcorr(template, data): 
    nhdr = template.shape[0]
    norm = np.array([get_norm(template=template[i], data=data[i]) for i in range(nhdr)])
    cc_sum = np.array([np.correlate(data[i], template[i])/norm[i] \
                   for i in range(nhdr)])
    return np.sum(cc_sum, axis=0)/nhdr

""" MPI env """
doublesize = MPI.DOUBLE.Get_size()
characsize = MPI.CHARACTER.Get_size()*8 #utf-8 encode with 8 bytes
nchar = 8 #headers will be up to nchar character

# -- all CPUs communicator
ALL_COMM = MPI.COMM_WORLD
rank, size = ALL_COMM.Get_rank(), ALL_COMM.Get_size()

# -- all CPUs inside node communicator
NODE_COMM = ALL_COMM.Split_type(MPI.COMM_TYPE_SHARED)
node_rank, node_size = NODE_COMM.rank, NODE_COMM.size

NODE_COMM.Barrier()

""" read arguments """
arglist = open(argv[1]).read().split()
data_path, template_path, year, julday = arglist

""" master nodes read data """
if node_rank == 0:
    if rank == 0:
        print("# -- Year: {}, Julian day: {}\n".format(year, julday))
        print('# -- Loading data...')
        t0 = time()
    st = read('{}/*{}.{}'.format(data_path, year, julday))
    st.detrend('demean')
    st.filter(type='bandpass', freqmin=2, freqmax=8)
    st.detrend('demean')
    starttime = md.date2num(st[0].stats.starttime.datetime)
    time_data = starttime + st[0].times()/24/3600
    if rank == 0:
      print('  -- data loaded in {:.02f}s: {} traces in Stream'.format(
          time()-t0, len(st)))
    dim = (len(st), np.min([tr.stats.npts for tr in st]))

    for r in range(1, node_size):
        NODE_COMM.send(dim, dest=r, tag=r)
        NODE_COMM.send(time_data, dest=r, tag=r)
else:
    dim = NODE_COMM.recv(source=0, tag=node_rank)
    time_data = NODE_COMM.recv(source=0, tag=node_rank)

NODE_COMM.Barrier()

""" place data and data headers in shared memory """
nhds, npts = dim
nbytes = nhds * npts * doublesize
wind = MPI.Win.Allocate_shared(nbytes if node_rank == 0 else 0, doublesize,
    comm=NODE_COMM)
bufd, doublesize = wind.Shared_query(0)
assert doublesize == MPI.DOUBLE.Get_size()
bufd = np.array(bufd, dtype='B', copy=False)
data = np.ndarray(buffer=bufd, dtype='d', shape=(nhds, npts))
if node_rank == 0:
   data.fill(0)

winh =  MPI.Win.Allocate_shared(nhds * characsize * nchar if node_rank == 0 else 0,
    characsize, comm=NODE_COMM)
bufh, doublesize = winh.Shared_query(0)
assert doublesize == characsize
bufh = np.array(bufd, dtype='B', copy=False)
data_header = np.ndarray(buffer=bufh, dtype='<U{}'.format(nchar), shape=nhds)

if node_rank == 0:
    for i, tr in enumerate(st):
        data[i] = tr.data[:npts] / np.sqrt(np.mean(tr.data[:npts]**2))
        data_header[i] = '{}.{}'.format(tr.stats.station, tr.stats.channel)
    if rank == 0:
        print('  -- data ready to use after {:.02f}s'.format(
          time()-t0))
    del st

NODE_COMM.Barrier()

""" list templates """
tmp_paths = np.sort(glob('{}/*.npz'.format(template_path)))
if rank == 0:
    print("# -- number of template found: {}".format(len(tmp_paths)))

""" distribute templates between all CPUs """
tmp_paths = tmp_paths[rank::size]

ALL_COMM.Barrier()

""" loop over templates """
for tmp_path in tmp_paths:

    """ load template """
    print("# CPU{} -- {}".format(rank, tmp_path.split('/')[-1]))
    tmp = np.load(tmp_path)
    npts_tmp = len(tmp['data'][0])
    moveout_tmp = tmp['moveout']

    """ select common stations between data and template """
    tmp_header = tmp['header']
    header = list(set(tmp_header).intersection(data_header))

    """ Format the input """
    Ncc = npts - npts_tmp + 1
    nhdr = len(header)
    moveout = np.zeros(nhdr, dtype=int)
    kdata = np.zeros(nhdr, dtype=int)
    template = np.zeros((nhdr, npts_tmp))
    for i, h in enumerate(header):
        k = tmp_header == h
        template[i] = tmp['data'][k]
        moveout[i] = moveout_tmp[k]
        kdata[i] = np.arange(len(data_header))[data_header == h]
    del k, moveout_tmp, header, tmp_header
    dmoveout = -np.max(moveout) + moveout

    """ matched-filter """
    print('# CPU{} -- start xcorr for T({}x{}) with D({}x{})'.format(rank, nhdr,
    npts_tmp, nhdr, npts))
    t0 = time()
    cc_sum = xcorr(template=template,
                   data=np.array([d[moveout[i]: len(d)+dmoveout[i]] \
                    for i, d in enumerate(data[kdata])]))
    del kdata, moveout, dmoveout, nhdr, npts_tmp, template
    print('# CPU{} -- xcorr done in {:.02f}s'.format(rank, time()-t0))

    """ Results """
    threshold = 8*mad(cc_sum)
    cc_vals = cc_sum[cc_sum >= threshold]
    tdetect = time_data[:len(cc_sum)][cc_sum >= threshold]
    Ndetect = len(cc_vals)

    if Ndetect > 0:
        out = 'Date {}.{}\n'.format(year, julday)
        for i in range(Ndetect):
            out += '{} {}\n'.format(md.num2date(tdetect[i]), cc_vals[i])
        fd = open('{}.log'.format('.'.join(tmp_path.split('.')[:-1])), 'a+')
        fd.write(out)
        fd.close()
        del out
    del cc_sum, threshold, cc_vals, tdetect, Ndetect
