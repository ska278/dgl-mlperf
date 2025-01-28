import os, sys
import torch as th
import torch.distributed as dist

def init_mpi(dist_backend, dist_url):
    world_size = -1
    if dist_backend == 'ccl':
        try:
            import oneccl_bindings_for_pytorch
        except:
            print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
            raise
    elif dist_backend == 'mpi':
        if not th.distributed.is_mpi_available():
            try:
                import torch_mpi
                print("imported torch_mpi.........")
            except:
                print("MPI backend requested but not available try installing torch_mpi module")
                raise
        else:
            raise ValueError(f"{dist_backend} backend requested but not supported")

    if dist_url == "env://" and world_size == -1:
        world_size = int(os.environ.get("PMI_SIZE", -1))
        if world_size == -1: world_size = int(os.environ["WORLD_SIZE"])


    distributed = world_size > 1
    if distributed:
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank == -1: rank = int(os.environ["RANK"])
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
    if rank == 0:
        print("Rank: ", rank ," World_size: ", world_size)
    return rank, world_size


class communicator:
    def __init__(self, rank, num_parts):
        self.rank, self.num_parts = rank, num_parts

    ## broadcasts and recvs a scalar or vector in tensor
    def communicate(self, sdata, async_op=False):
        assert th.is_tensor(sdata) == True
        dlen = sdata.shape[0]
        assert dlen == 1
        scount = [dlen for i in range(self.num_parts)]
        rcount = [dlen for i in range(self.num_parts)]

        tot_size = th.sum(rcount)
        #sdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        sdata_msg = th.cat([sdata]*self.num_parts, dim=0)
        rdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        assert sdata_msg.shape[0] == rdata_msg.shape[0]
        req = dist.all_to_all_single(rdata_msg, sdata_msg,
                                     rcount, scount, async_op=True)
        if async_op:
            return req, sdata_msg, rdata_msg
            
        req.wait()
        return sdata_msg, rdata_msg

    
    ## all2all single elements
    def communicatev(self, sdata, async_op=False):
        assert th.is_tensor(sdata) == True
        dlen = 1
        assert sdata.shape[0]== self.num_parts
        scount = [dlen for i in range(self.num_parts)]
        rcount = [dlen for i in range(self.num_parts)]

        tot_size = th.sum(rcount)
        #sdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        sdata_msg = sdata ## th.cat([sdata]*self.num_parts, dim=0)
        rdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        assert sdata_msg.shape[0] == rdata_msg.shape[0]
        req = dist.all_to_all_single(rdata_msg, sdata_msg,
                                     rcount, scount, async_op=True)
        if async_op:
            return req, sdata_msg, rdata_msg
            
        req.wait()
        return sdata_msg, rdata_msg
    

    ## broadcasts and recvs a scalar or vector in tensor or list
    def communicate_any(self, sdata, datatype, async_op=False):
        if not th.is_tensor(sdata):
            sdata = th.tensor(sdata, dtype=datatype)

        dlen = sdata.shape[0]
        scount = [dlen for i in range(self.num_parts)]
        rcount = [dlen for i in range(self.num_parts)]

        tot_size = sum(rcount)
        # sdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        sdata_msg = th.cat([sdata]*self.num_parts, dim=0)
        rdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        assert sdata_msg.shape[0] == rdata_msg.shape[0]
        req = dist.all_to_all_single(rdata_msg, sdata_msg,
                                     rcount, scount, async_op=True)
        if async_op:
            return req, sdata_msg, rdata_msg
        
        req.wait()
        return sdata_msg, rdata_msg

    
    def comv(self, scount):
        if not th.is_tensor(scount):
            scount = th.tensor(sdata, dtype=th.int32)
        rcount = th.empty(self.num_parts, dtype=th.int32)
        s = [1 for i in range(self.num_parts)]
        r = s
        dist.all_to_all_single(rcount, scount, r, s)
        return scount, rcount

    
    def all2allv(self, scount, send_nodes, async_op=False):
        assert th.is_tensor(send_nodes) == True
        if not th.is_tensor(scount):
            #send_sr, recv_sr = self.communicate_any(scount, th.int32)
            scount = th.tensor(scount, dtype=th.int32)
        #else:
            #send_sr, recv_sr = self.communicate(scount)

        send_sr, recv_sr = self.comv(scount)
        trecv = th.sum(recv_sr)
        #tsend, trecv = th.sum(send_sr), th.sum(recv_sr)  ##recv data
        recv_nodes = th.empty(trecv, dtype=send_nodes.dtype)
        rcount = recv_sr.tolist()
        scount = send_sr.tolist()
        #print('[all2allv] rcount: ', rcount)
        #print('[all2allv] scount: ', scount)
        req = dist.all_to_all_single(recv_nodes, send_nodes,
                                     rcount, scount,
                                     async_op=True)
        if async_op:
            return req, recv_nodes, rcount
        
        req.wait()
        return recv_nodes, rcount

    
    ## braodcasts and recvs a tensor
    def all2all(self, sdata, async_op=False):
        #def communicate_vector(self, sdata):   
        assert th.is_tensor(sdata) == True
        dlen = sdata.shape[0]
        if dlen > 1:
            #scount = [dlen] # for i in range(self.num_parts)]
            scount = [dlen for i in range(self.num_parts)]
            #scount, rcount = self.communicate_any(scount, th.int32)
            scount, rcount = self.comv(scount)
            tot_size = th.sum(rcount)
        else:
            scount = [dlen for i in range(self.num_parts)]
            rcount = [dlen for i in range(self.num_parts)]
            tot_size = sum(rcount)            

        #sdata_msg = th.empty(tot_size, dtype=sdata.dtype)
        sdata_msg = th.cat([sdata]*self.num_parts, dim=0)
        rdata_msg = th.empty(tot_size, dtype=sdata.dtype)
            
        #assert sdata_msg.shape[0] == rdata_msg.shape[0]
        req = dist.all_to_all_single(rdata_msg, sdata_msg,
                                     rcount, scount, async_op=True)
        if async_op:
            return req, rdata_msg, rcount

        req.wait()
        return rdata_msg, rcount
