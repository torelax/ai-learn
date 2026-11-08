import time
import numpy as np

class Clock:

    """
    A class used to keep track of elapsed time.

    Args:
        duration (int, optional): The initial duration in seconds. Default is 0.

    Attributes:
        duration (int): The total elapsed duration in seconds.
        current (float): The current time in seconds since the poch (UTC).
    """

    def __init__(self, duration=0):
        self.duration = duration
        self.current = None

    def start(self):

        """
        Starts the clock.
        """

        # init clocking
        self.current = time.time()

    def clock(self, start, end):

        """
        Finds the duration between the start and end times.

        Args:
            start (float): The start time in seconds.
            end (float): The end time in seconds.

        Returns:
            Tuple[float, float, float]: The elapsed time in hours, minutes, and seconds.
        """

        # find duration between start & end
        elapsed = end - start
        self.current = end
        self.duration += elapsed
        return self.to_hour_min_sec(elapsed)
    
    def tick(self):

        """
        Finds the time between the current time and the last epoch.

        Returns:
            str: The elapsed time in the format 'HH:MM:SS'.
        """

        # verify the clock has started
        if self.current is None:
            raise ValueError("The clock has not started. Please call Clock.start().")

        # find time between then & now
        now = time.time()
        h, m, s = self.clock(self.current, now)
        return self.asstr(h, m, s)

    def elapsed(self):

        """
        Gets the total elapsed time since the clock started.

        Returns:
            str: The elapsed time in the format 'HH:MM:SS'.
        """

        # get time clock has been ticking
        elapsed = self.duration
        h, m, s = self.to_hour_min_sec(elapsed)
        return self.asstr(h, m, s)
        
    def reset(self):
        
        """
        Resets the clock to the initial state.
        """

        self.__init__(duration=0)

    def to_hour_min_sec(self, elapsed):

        """
        Converts elapsed seconds to hours, minutes, and seconds.

        Args:
            elapsed (float): The elapsed time in seconds.

        Returns:
            Tuple[float, float, float]: The elapsed time in hours, minutes, and seconds.
        """

        # convert elapsed seconds to hours, min, & seconds
        hours, rem = elapsed // 3600, elapsed % 3600
        minutes, seconds = rem // 60, rem % 60
        return hours, minutes, seconds
    
    def asstr(self, hours, minutes, seconds):

        """
        Formats hours, minutes, and seconds as a string.

        Args:
            hours (float): The elapsed hours.
            minutes (float): The elapsed minutes.
            seconds (float): The elapsed seconds.

        Returns:
            str: The elapsed time in the format 'HH:MM:SS'.
        """

        # format for strings
        hstr = f"0{hours:.0f}" if np.rint(hours) < 10 else f"{hours:.0f}"
        mstr = f"0{minutes:.0f}" if np.rint(minutes) < 10 else f"{minutes:.0f}"
        sstr = f"0{seconds:.0f}" if np.rint(seconds) < 10 else f"{seconds:.0f}"
        return f"{hstr}:{mstr}:{sstr}"