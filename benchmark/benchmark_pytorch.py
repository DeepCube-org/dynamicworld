from benchmark import Benchmark
import torch

import segmentation_models_pytorch as smp

class PyTorchBenchmark(Benchmark):

    def get_dummy(self, shape):
        x = torch.randn(shape[0], shape[1], shape[2], shape[3], dtype=torch.float, device=self.device)
        return(x)
    
    def load_model(self, path=None):
        print('path parameter is not used')
        
        self.device = torch.device("cuda")
        self.model = smp.Unet(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                # model output channels (number of classes in your dataset)
        )
        self.model = self.model.to(self.device)
        
        import torch._dynamo
        torch._dynamo.reset()
        self.model = torch.compile(self.model, mode = 'reduce-overhead')

        print('PyTorch Version:', torch.__version__)
        print('Device:',self.device)
        

    def time_model(self, model, dummy_input):
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        return(curr_time)