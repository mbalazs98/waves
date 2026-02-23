
import numpy as np
import mnist
from ml_genn.utils.data import preprocess_spikes



class MovingMNISTDataset():
    """
    Dataset for generating sequences of moving MNIST digits with wrap-around boundary conditions.

    Args:
        train (bool): if True, load training split; otherwise, test split.
        seq_len (int): number of frames in each sequence.
        image_size (int): height and width of the square output frames.
        velocity_range_x (tuple of int): (min_vx, max_vx) inclusive range for horizontal velocity.
        velocity_range_y (tuple of int): (min_vy, max_vy) inclusive range for vertical velocity.
        num_digits (int): number of MNIST digits to overlay in each sequence.
        transform (callable, optional): transform applied to the full sequence tensor.
        seed (int): random seed for deterministic selection.
    """
    def __init__(
        self,
        train=True,
        seq_len=20,
        image_size=28,
        velocity_range_x=(-3, 3),
        velocity_range_y=(-3, 3),
        num_digits=2,
        seed=42
    ):
        super().__init__()
        mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        self.mnist = [(x, y) for x, y in zip(mnist.train_images(),mnist.train_labels())] if train else [(x, y) for x, y in zip(mnist.test_images(),mnist.test_labels())]
        
        self.seq_len = seq_len
        self.image_size = image_size
        self.vx_min, self.vx_max = velocity_range_x
        self.vy_min, self.vy_max = velocity_range_y
        self.num_digits = num_digits
        self.rng = np.random.RandomState(seed)


    def generate(self):
        # Sample digits and their labels
        imgs = []
        labels = []
        for n in range(self.num_digits):
            idx = self.rng.randint(0, len(self.mnist))
            img, lbl = self.mnist[idx]
            imgs.append(img)
            labels.append(lbl)

        # Sample integer velocities and initial positions for each digit
        velocities_x = []
        velocities_y = []
        positions_x = []
        positions_y = []
        
        # Prepare output sequence
        '''seq = torch.zeros(
            self.seq_len, 1, self.image_size, self.image_size, dtype=imgs[0].dtype
        )'''
        times = []
        inds = []

        for _ in range(self.num_digits):
            # Sample integer velocities for this digit
            vx = self.rng.randint(self.vx_min, self.vx_max + 1)
            vy = self.rng.randint(self.vy_min, self.vy_max + 1)
            velocities_x.append(vx)
            velocities_y.append(vy)
            
            # Sample integer initial positions (top-left corner of digit)
            x0 = self.rng.randint(0, self.image_size)
            y0 = self.rng.randint(0, self.image_size)
            positions_x.append(x0)
            positions_y.append(y0)

        # Generate each frame by rolling each digit and summing
        for t in range(self.seq_len):
            frame = np.zeros((self.image_size, self.image_size))
            for i, img in enumerate(imgs):
                '''# Pad the MNIST image (28x28) to match the frame size if needed (image_size x image_size)
                pad_size = (self.image_size - img.shape[1]) // 2
                padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
                
                # If padding is uneven, add extra padding to the right/bottom
                if padded_img.shape[1] < self.image_size:
                    padded_img = F.pad(padded_img, (0, self.image_size - padded_img.shape[2], 0, self.image_size - padded_img.shape[1]), "constant", 0)
                
                # Replace the original image with the padded version
                img = padded_img'''
                
                # Calculate current position based on this digit's velocity and initial position
                shift_x = positions_x[i] + velocities_x[i] * t
                shift_y = positions_y[i] + velocities_y[i] * t
                
                # Wrap-around translation via torch.rollyw
                moved = np.roll(
                    img,
                    shift=shift_x,#(shift_y, shift_x),  # (vertical, horizontal)
                    axis=(0, 1),
                )
                frame += moved

            # Clamp to [0, 1] in case of overlap
            active = np.where(frame.flatten()>0.9)
            for a in active[0]:
                inds.append(a)
                times.append(t)

        if self.num_digits == 1:
            return preprocess_spikes(np.asarray(times), np.asarray(inds), self.image_size*self.image_size), labels[0]
        else:
            return preprocess_spikes(np.asarray(times), np.asarray(inds), self.image_size*self.image_size), np.asarray(labels)


class FixedVelocityMovingMNIST(MovingMNISTDataset):
    """Same as MovingMNISTDataset, but every digit moves at a fixed (vx, vy)."""
    def __init__(self, vx, vy, **kwargs):
        super().__init__(velocity_range_x=(vx, vx),
                         velocity_range_y=(vy, vy),
                         **kwargs)



