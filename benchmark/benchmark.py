import numpy as np



class Benchmark:


    def __init__(
        self, 
        path,
        resolution = 224,
        channels = 3
    ):
        self.resolution = resolution
        self.channles = channels
        self.load_model(path)

    def load_model(self, path):
        pass

    def get_dummy(self, shape):
        """
        Get a dummy variable of a given shape
        """
        pass
    
    def time_model(self, model, dummy_input):
        """
        Get the time spent for the inference, measured in ms
        """
        pass

    def warm_up(self, model, repetitions=50):
        dummy_input = self.get_dummy((1, self.resolution, self.resolution, self.channels)) 
        for _ in range(repetitions):
            _ = self.time_model(model, dummy_input)

    def get_optimal_resolution(self, model):
        self.warm_up(model)
        optimal_resolution = 128
        for resolution in [256, 512, 1024, 2048, 4096]:
            dummy_input = self.get_dummy((1, resolution, resolution, self.channels))
            try:
                _ = self.time_model(model, dummy_input)
                optimal_resolution = resolution
            except RuntimeError as e:
                print(e)
                break
        return(optimal_resolution)

    def get_latency(self, model, batch_size, resolution):
        self.warm_up(model)

        repetitions = 300
        timings=np.zeros((repetitions,1))

        # MEASURE PERFORMANCE
        for rep in range(repetitions):
            dummy_input = self.get_dummy((batch_size, resolution, resolution, self.channels))
            timings[rep] = self.time_model(model, dummy_input)

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)    
        return mean_syn, std_syn


    def get_optimal_batch_size(self, model):
        self.warm_up(model)

        optimal_batch_size = 1
        for batch_size in [32, 64, 128, 256, 512, 1024]:
            dummy_input = self.get_dummy((batch_size, self.resolution, self.resolution, self.channels))
            try:
                _ = model(dummy_input)
                optimal_batch_size = batch_size
            except RuntimeError as e:
                print(e)
                break
        return(optimal_batch_size)


    def get_throughput(self, model, batch_size, resolution):
        self.warm_up(model)

        repetitions = 100
        total_time  = 0
        for rep in range(repetitions):
            dummy_input = self.get_dummy((batch_size, resolution, resolution, self.channels))
            total_time += self.time_model(model, dummy_input)/1000 #to convert in second (original in ms)

        throughput =   (repetitions*batch_size)/total_time  # n_images/total_time 
        return(throughput)


    def metrics(
        self,
        latency_batch_size,
        throughput_batch_size
    ):

        for _ in range(2):
            mean, std = self.get_latency(self.model, latency_batch_size, self.resolution)

        print('Latency, average time (ms):', mean)
        print('Latency, std time (ms):', std)

        #optimal_batch_size = get_optimal_batch_size(model)
        optimal_batch_size = throughput_batch_size
        for _ in range(2):
            throughput = self.get_throughput(self.model, optimal_batch_size, self.resolution)

        print('Optimal Batch Size:', optimal_batch_size)
        print('Final Throughput (imgs/s):',throughput)
