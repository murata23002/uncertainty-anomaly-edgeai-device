import cv2

from logger.custom_logger import custom_logger


class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.cap = None
        self.width = width
        self.height = height

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            custom_logger.error(f"Failed to open camera with index {self.camera_index}")
            raise RuntimeError(
                f"Camera with index {self.camera_index} could not be opened."
            )

        # Set camera width and height if specified
        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            custom_logger.info(f"Set camera width to {self.width}")

        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            custom_logger.info(f"Set camera height to {self.height}")

        custom_logger.info(f"Camera {self.camera_index} initialized successfully.")
        return self

    def get_frame(self):
        if self.cap is None:
            custom_logger.error("Capture device not initialized.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            custom_logger.error("Failed to retrieve frame from camera.")
            return None
        return frame

    def get_camera_size(self):
        """Get the current camera frame size."""
        if self.cap is None:
            custom_logger.error("Capture device not initialized.")
            return None, None

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        custom_logger.info(f"Current camera size: width={width}, height={height}")
        return width, height

    def release(self):
        if self.cap:
            self.cap.release()
            custom_logger.info(f"Camera {self.camera_index} released.")

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        if exc_type:
            custom_logger.error(f"An error occurred: {exc_value}")

    def iterate_frames(self, flip=False):
        while True:
            frame = self.get_frame()
            if frame is None:
                custom_logger.warning("No frame captured, stopping.")
                break
            if flip:
                frame = cv2.flip(frame, 1)
            yield frame
