import time

class Benchmark:
    def __init__(self, id, enable, visible, except_first, show_callback=None):
        self.id = id
        self.elapsed_times = []
        self.first = True
        self.except_first = except_first
        self.enable = enable
        self.visible = visible
        self.show_callback = show_callback

        self.newest_mean_time = None

    def reset(self):
        self.elapsed_times = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        elapsed_time = time.time() - self.start_time

        if self.first:
            self.first = False
            if not self.except_first:
                self.elapsed_times.append(elapsed_time)
        else:
            self.elapsed_times.append(elapsed_time)
        
        return elapsed_time

    def get_mean(self):
        n = len(self.elapsed_times)
        if n > 0:
            return sum(self.elapsed_times) / n
        else:
            return None

    def get_fps(self):
        mean_time = self.get_mean()
        if mean_time is None:
            return None
        else:
            return 1.0 / mean_time

    def get_result(self):
        elapsed_time = self.end()
        mean_time = self.get_mean()
        
        self.newest_mean_time = mean_time

        if mean_time is None:
            return "[{}] {:.1f}ms / mean ---".format(self.id, elapsed_time * 1000)
        else:
            return "[{}] {:.1f}ms / mean {:.1f}ms".format(self.id, elapsed_time * 1000, mean_time * 1000)

    def show(self):
        if self.enable:
            if self.show_callback is None:
                bench_str = self.get_result()
                if self.visible:
                    print(bench_str)
            else:
                self.show_callback(self.get_result())

    def show_fps(self):
        if self.visible:
            fps = self.get_fps()
            if fps is not None:
                msg = "[{}] {:.1f}FPS".format(self.id, fps)
                if self.show_callback is None:
                    print(msg)
                else:
                    self.show_callback(msg)

    @staticmethod
    def create_timers(id_list, except_first=False, enable=True, visible=True, show_callback=None):
        timers = {}

        for id in id_list:
            timers[id] = Benchmark(id, except_first=except_first, enable=enable, visible=visible, show_callback=show_callback)

        return timers