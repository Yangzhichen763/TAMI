import time


def set_proctitle_progress(proctitle, progress=None, rolling=True, num_blank=2, proctitle_max_width=19, rolling_speed=1.0):
    try:
        import setproctitle
    except ImportError as _:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'setproctitle'])
        import setproctitle

    pos = int(time.time())

    # display progress bar as a rolling caption
    if rolling:
        full_proctitle = proctitle + ' ' * num_blank

        proctitle = ''
        for i in range(proctitle_max_width):
            k = (int(pos * rolling_speed + i)) % len(full_proctitle)
            proctitle += full_proctitle[k]
        proctitle = f"|{proctitle}|"

    # set proctitle with progress percentage
    if progress is None:
        setproctitle.setproctitle(proctitle)
    elif isinstance(progress, str):
        progress = progress.strip()[:6]
        setproctitle.setproctitle(f"{proctitle}:{progress}")
    elif isinstance(progress, float) and 0 <= progress <= 1:
        setproctitle.setproctitle(f"{proctitle}:{progress * 100:.2f}%")
    else:
        raise ValueError("progress should be a string or a float between 0 and 1")


#region ==[Proctitle Context]==
import threading

class ProctitleContext:
    # one instance for one process (and all threads in one process)
    singleton = None

    def __init__(
            self, proctitle, default_label=None,
            rolling=True, num_blank=2, proctitle_max_width=18,
            update_interval=1,
    ):
        if ProctitleContext.singleton is not None:
            if default_label is not None:
                self.set_label(default_label)
            else:
                self.set_label(proctitle)
            return

        ProctitleContext.singleton = self

        self.proctitle = proctitle
        self.rolling = rolling
        self.num_blank = num_blank
        self.proctitle_max_width = proctitle_max_width
        self.update_interval = update_interval

        self.stop = False
        self.thread = None

        self.labels = {}
        self.default_label = default_label
        self.progress = None

    def __enter__(self):
        if self is not ProctitleContext.singleton:
            return self

        self.thread = threading.Thread(target=self.update_proctitle, args=())
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self is not ProctitleContext.singleton:
            if id(self) in ProctitleContext.singleton.labels:
                del ProctitleContext.singleton.labels[id(self)]
                return

        self.stop = True
        self.thread.join()

    def set_progress(self, progress=None):
        if self is not ProctitleContext.singleton:
            return self

        self.progress = progress
        if not self.thread.is_alive():  # if thread has been killed, force update
            set_proctitle_progress(self.proctitle, self.progress, self.rolling, self.num_blank, self.proctitle_max_width)

    def set_label(self, label):
        if self is not ProctitleContext.singleton:
            ProctitleContext.singleton.labels[id(self)] = label

    def update_proctitle(self):
        i_labels = 0
        while not self.stop:
            any = False
            if self.progress is not None:
                set_proctitle_progress(self.proctitle, self.progress, self.rolling, self.num_blank, self.proctitle_max_width)
                time.sleep(self.update_interval)
                any = True

            if len(self.labels) == 0 and self.default_label is not None:
                set_proctitle_progress(self.proctitle, self.default_label, self.rolling, self.num_blank, self.proctitle_max_width)
                time.sleep(self.update_interval)
                any = True
            # update one label at a time
            elif len(self.labels) > 0:
                try:
                    i_labels = (i_labels + 1) % len(self.labels)

                    label = list(self.labels.values())[i_labels]
                    set_proctitle_progress(self.proctitle, label, self.rolling, self.num_blank, self.proctitle_max_width)
                    time.sleep(self.update_interval)
                    any = True
                except Exception as e:
                    pass

            if not any:
                time.sleep(self.update_interval)

#endregion


if __name__ == '__main__':
    import time
    import argparse
    import setproctitle

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', type=str, default='Banner message')
    parser.add_argument('--gpus', '-g', type=str, default='0', help='gpus to use')
    parser.add_argument('--width', '-w', type=int, default=19, help='max width of the banner message')
    parser.add_argument('--speed', '-v', type=float, default=5, help='speed of the banner message')
    args = parser.parse_args()

    set_proctitle_progress("utils for rolling message display", num_blank=2, proctitle_max_width=args.width)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    import torch

    test_tensor = torch.tensor([0], device='cuda')
    t_delta = 1 / args.speed

    while True:
        set_proctitle_progress(args.text, num_blank=2, proctitle_max_width=args.width, rolling_speed=args.speed)
        test_tensor += test_tensor  # force update on gpu
        time.sleep(t_delta)