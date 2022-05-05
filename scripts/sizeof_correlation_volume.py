from socket import SO_VM_SOCKETS_BUFFER_MIN_SIZE


if __name__ == "__main__":
    import torch, sys

    def calculate_sizeof_correlation_volume(h, w):
        
        x = torch.ones((h,w,h,w), dtype=torch.float32, requires_grad=True)
        
        nbytes = sys.getsizeof(x.storage())
        nbytes_manual = x.nelement()*x.element_size()
        
        print(f"Correlation volume requires during training: {nbytes/10e6} mb ~= {nbytes_manual/10e6} mb")


    WD = 400
    HT = 400

    calculate_sizeof_correlation_volume(HT, WD)