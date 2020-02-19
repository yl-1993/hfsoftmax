# -*- coding: utf-8 -*-

import zmq
import json
import h5py
import logging
import threading
import numpy as np


def init_gaussian(shape, order='C'):
    arr = 0.01 * np.random.randn(*shape).astype(np.float32)
    if order == 'F':
        return np.asfortranarray(arr)
    else:
        return arr


def init_uniform(shape, low=-0.1, high=0.1, order='C'):
    assert low < high
    arr = np.random.uniform(low, high, shape).astype(np.float32)
    if order == 'F':
        return np.asfortranarray(arr)
    else:
        return arr


def init_zeros(shape, order='C'):
    return np.zeros(shape, order=order).astype(np.float32)


class Optim(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ParameterServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5570')

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        worker = ParameterWorker(context)
        worker.start()

        try:
            zmq.proxy(frontend, backend)
        except zmq.ContextTerminated:
            frontend.close()
            backend.close()


class ParameterWorker(threading.Thread):
    def __init__(self, context, lr=0.01, weight_decay=1e-4, momentum=0.9):
        threading.Thread.__init__(self)
        self.context = context
        self.clients = {}
        self.grads = {}
        self.mtable = {}
        self.key, self.wkey, self.hkey = ['@meta@ps', 'w', 'h']
        self.updater = 'sgd'
        self.support_keys = [self.wkey, self.hkey]
        assert lr > 0
        assert weight_decay >= 0
        assert momentum >= 0
        self.optim = Optim()
        self.optim.lr = lr
        self.optim.weight_decay = weight_decay
        self.optim.momentum = momentum

    def run(self):
        self._socket = self.context.socket(zmq.DEALER)
        self._socket.connect('inproc://backend')
        print('Worker started')
        while True:
            self._recv()

        self._socket.close()
        # since `proxy` is daemon
        # we have to teminate it in a not graceful way
        if self.context:
            print('Terminate proxy ... ')
            time.sleep(1.)  # make sure proxy pass the last msg to clients
            self.context.term()

    """ comm
    """
    @staticmethod
    def _parse_json(x):
        return json.loads(x.decode('utf-8'))

    @staticmethod
    def _buf_to_ndarray(x, md):
        x = np.frombuffer(x, dtype=md['dtype'])
        return x.reshape(md['shape'])

    def _ready_for_update(self, mid):
        # TODO: following neglect the situation that a client sends the request twice
        for k in self.clients:
            if self.clients[k][mid] == 0:
                return False
        return True

    def _recv(self):
        # TODO:
        #   1. use logging to better log server info
        #   2. better exception handling
        print('waiting for message ... ')
        packet = self._socket.recv_multipart()
        if len(packet) == 2:
            ident, msg = packet
            msg = self._parse_json(msg)
        elif len(packet) == 4:
            ident, msg, meta, data = packet
            msg, meta = map(self._parse_json, [msg, meta])
            if msg['op'] == 'set_matrix':
                msg['data'] = self._buf_to_ndarray(data, meta)
            else:
                msg['rows'] = self._buf_to_ndarray(data, meta)
        elif len(packet) == 6:
            ident, msg, rows_meta, rows_data, val_meta, val_data = packet
            msg, rows_meta, val_meta = map(self._parse_json,
                                           [msg, rows_meta, val_meta])
            msg['rows'] = self._buf_to_ndarray(rows_data, rows_meta)
            msg['data'] = self._buf_to_ndarray(val_data, val_meta)
        else:
            raise RuntimeError('Unsupported msg type')
        self.handle(ident, msg)
        # try:
        #     self.handle(ident, msg)
        # except Exception as err:
        #     print('Error ', err)

    def _recv_array(self, flags=0, copy=True, track=False):
        ident = self._socket.recv_json(flags=flags)
        md = self._socket.recv_json(flags=flags)
        print("recv a numpy array {}".format(md))
        msg = self._socket.recv(flags=flags, copy=copy, track=track)
        data = np.frombuffer(msg, dtype=md['dtype'])
        return data.reshape(md['shape'])

    def _send(self, ident, data):
        assert isinstance(data, np.ndarray)
        meta = {'dtype': str(data.dtype), 'shape': data.shape}
        self._socket.send(ident, zmq.SNDMORE)
        self._socket.send_json(meta)
        self._socket.send_multipart([ident, data])

    """ public functions
    """
    def handle(self, ident, msg):
        op = msg['op']
        print('receiving {} from {}'.format(op, ident))
        if op == 'register':
            self.clients[ident] = {}
            print('Client {} register. ({} clients in total).'.format(
                ident, len(self.clients)))
        elif op == 'exit':
            del self.clients[ident]
            print('Client {} exit. ({} clients in total).'.format(
                ident, len(self.clients)))
        elif op == 'add_matrix':
            mid = msg['mid']
            self.add_matrix(mid, msg['shape'], init_uniform)
            self._reset_grad(mid)
        elif op == 'set_matrix':
            assert 'data' in msg
            force = False
            if 'force' in msg and msg['force']:
                force = True
            self.set_matrix(msg['mid'], msg['data'], force)
            print('the value of matrix {} has been set.'.format(msg['mid']))
        elif op == 'get_value_by_rows':
            weights = self.get_value_by_rows(msg['mid'], msg['rows'])
            # send back to client
            self._send(ident, weights)
        elif op == 'set_value_by_rows':
            assert 'data' in msg
            self.set_value_by_rows(msg['mid'], msg['rows'], msg['data'])
            print('the value of {} rows of matrix {} has been set.'.format(
                len(msg['rows']), msg['mid']))
        elif op == 'update_params':
            self.update_params(msg)
        elif op == 'update_by_rows':
            assert 'data' in msg
            mid = msg['mid']
            # merge data from all clients
            assert len(msg['rows']) == len(msg['data'])
            self.clients[ident][mid] = 1
            for i, r in enumerate(msg['rows']):
                if r not in self.grads[mid]:
                    self.grads[mid][r] = np.array(msg['data'][i])
                else:
                    self.grads[mid][r] += np.array(msg['data'][i])
            if self._ready_for_update(mid):
                print('updating')
                skip_decay = False
                if len(self.clients) == 1 and \
                    'skip_decay' in msg and msg['skip_decay']:
                    print('skipping weight decay')
                    skip_decay = True
                self.update_by_rows(mid,
                                    np.array(list(self.grads[mid].keys())),
                                    np.array(list(self.grads[mid].values())),
                                    skip_decay=skip_decay)
                print("weight change", self.mtable[mid][self.wkey].mean())
                # reset gradient
                self._reset_grad(mid)
        elif op == 'snapshot':
            self.snapshot(msg['path'])
        elif op == 'load':
            self.load(msg['path'])
        elif op == 'resume':
            self.resume(msg['path'])
        else:
            raise KeyError('Unknown operation')

    def add_matrix(self, mid, shape, init_func, his=True):
        mid = self._build_mtable(mid)
        # TODO: add `force` to rm already built matrix
        if mid is None:
            return
        self.mtable[mid][self.wkey] = init_func(shape)
        print(self.mtable[mid][self.wkey].shape)
        if his:
            self.mtable[mid][self.hkey] = init_zeros(shape)
        self._check_order(mid)

    def load_matrix(self, mid, h5group, his=True):
        mid = self._build_mtable(mid)
        if mid is None:
            return
        for key in h5group:
            if not key in self.support_keys:
                raise KeyError('The {} is not in the support list'.format(key))
            self.mtable[mid][key] = np.asfortranarray(h5group[key])
            if key == self.hkey and not his:
                self.mtable[mid][key].fill(0)
        self._check_order(mid)

    def get_value_by_rows(self, mid, rows):
        return self.mtable[mid][self.wkey][rows, :]

    def set_matrix(self, mid, data, force=False):
        """ Note that when you set value of weights directly,
            the history of SGD will automatically be set to zero.
            If `force` is not true, the shape of input data
            should be equal to the shape of existing weights.
        """
        if not force:
            assert data.shape == self.mtable[mid][self.wkey].shape
        self.mtable[mid][self.wkey][:] = data
        if self.hkey in self.mtable[mid]:
            self.mtable[mid][self.hkey].fill(0)

    def set_value_by_rows(self, mid, rows, data):
        """ Note that when you set value of weights directly,
            the history of SGD will automatically be set to zero.
        """
        self.mtable[mid][self.wkey][rows, :] = data
        if self.hkey in self.mtable[mid]:
            self.mtable[mid][self.hkey][rows, :].fill(0)

    def update_by_rows(self, mid, rows, grad, skip_decay=False):
        """ Note that the gradient from PyTorch is already conducted L2 regularization!
            That is, $grad += weight * weight\_decay$ has been applied to the grad.
            If you use `param.grad` from PyTorch Parameter, you don't have to regularize
            weights again.
        """
        if self.optim.weight_decay > 0:
            if not skip_decay:
                grad = self._l2_regularize(
                    self.mtable[mid][self.wkey][rows, :], grad)
            pass
        self._sgd_update(mid, rows, grad)

    def update_params(self, msg):
        for key in msg:
            if key == 'op':
                pass
            elif key not in self.optim:
                raise KeyError('Not supported key found: {}'.format(key))
            else:
                val = msg[key]
                if key == 'lr':
                    assert val > 0
                else:
                    assert val >= 0
                self.optim[key] = val
                print("{} has been updated to {}".format(key, val))

    def snapshot(self, path):
        with h5py.File(path, 'w') as f:
            print('snapshot to {}'.format(path))
            ps = f.create_group(self.key)
            for key in self.mtable:
                midg = ps.create_group(str(key))
                for k in self.mtable[key]:
                    midg[k] = self.mtable[key][k][...]

    def resume(self, path, his=True):
        with h5py.File(path, 'r') as f:
            print('resume from {} with history={}'.format(path, his))
            if self.key not in f.keys():
                logging.warn('The model does not have {}'.format(self.key))
                return
            ps = f[self.key]
            for key in ps.keys():
                self.load_matrix(key, ps[key], his)

    def load(self, path):
        self.resume(path, his=False)

    """ private functions
    """
    def _exists(self, mid):
        return (mid in self.mtable)

    def _build_mtable(self, mid):
        if isinstance(mid, str):
            pass
        elif isinstance(mid, int):
            mid = str(mid)
        elif isinstance(mid, unicode):
            mid = mid.encode('ascii', 'ignore')
        else:
            raise TypeError(
                'The key({},{}) for Parameter Server should be str!'.format(
                    mid, type(mid)))
        if not self._exists(mid):
            self.mtable[mid] = {}
            return mid
        else:
            return None

    def _check_order(self, mid):
        for key in self.mtable[mid]:
            if not self.mtable[mid][key].flags['C_CONTIGUOUS']:
                raise TypeError('np.darray should be C order!')

    def _reset_grad(self, mid):
        for k in self.clients:
            self.clients[k][mid] = 0
        self.grads[mid] = {}

    def _l2_regularize(self, data, grad):
        grad += self.optim.weight_decay * data
        return grad

    def _sgd_update(self, mid, rows, grad):
        """ The algorithm here is compatible with PyTorch.
            Note that it is different from platform like Caffe.
            Detailed description can be found at
            https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
        """
        if self.optim.momentum > 0:
            grad += self.mtable[mid][self.hkey][rows, :] * self.optim.momentum
            self.mtable[mid][self.hkey][rows, :] = grad
        self.mtable[mid][self.wkey][rows, :] -= self.optim.lr * grad


if __name__ == "__main__":
    ps = ParameterServer()
    ps.start()
