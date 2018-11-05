# -*- coding: utf-8 -*-

""" Module summary description.

More detailed description.
"""

# __all__ = []
# __version__ = '0.1'
import sys
import time
from functools import wraps

import progressbar as pb

__author__ = 'Benjamin Pillot'
__copyright__ = 'Copyright 2018, Benjamin Pillot'
__email__ = 'benjaminpillot@riseup.net'


# Decorator for triggering event
def broadcast_event(description, message_level=2, no_time=False):
    def decorate(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            trigger_event("update_description", description, message_level, no_time)
            if no_time:
                output = func(*args, **kwargs)
            else:
                with Timer() as t:
                    output = func(*args, **kwargs)
                trigger_event("update_time", t)
            return output
        return wrapper
    return decorate


def event(name):
    # on assure que la liste des events et callabcks est initialisae
    event.broadcast = getattr(event, 'broadcast', {})

    def decorate(func):
        # on ajoute la fonction comme callback pour cet event
        event.broadcast.setdefault(name, []).append(func)

        return func

    return decorate


def trigger_event(name, description, level=None, no_time=None):
    """ Trigger event with respect to its name

    If no progress display is called, do nothing (AttributeError --> pass)
    :param name:
    :param description:
    :param level:
    :param no_time:
    :return:
    """
    try:
        broadcast = getattr(event, "broadcast")
        for f in broadcast[name]:
            f(description, level, no_time)
    except AttributeError:
        pass


def progress(max_value, start_description="Initiating", description_max_size=30, carriage_return=False):

    progress.step = getattr(progress, "step", 0)
    progress_bar = ProgressBar(max_value, start_description, description_max_size, carriage_return)

    @event("update_description")
    def update_description(description):
        progress_bar.update(progress.step, description)

    @event("update_value")
    def update_value(description):
        progress.step += 1
        progress_bar.update(progress.step, description)


def display_progress(description_max_size=50, description_title="Running job", time_title="Elapsed time"):

    print(description_title + (description_max_size - len(description_title)) * " " + "| " + time_title)
    display_progress.step = 0

    @event("update_description")
    def print_description(description, level, no_time):
        description = "  " * level + description
        if no_time:
            separator = ""
        else:
            separator = "| "
        display_progress.description = description + (description_max_size - len(description)) * " " + separator
        if display_progress.step == 0:
            write(display_progress.description)
            display_progress.step += 1
        else:
            write("\n" + display_progress.description)

    @event("update_time")
    def print_time(time, *args):
        delete()
        write(display_progress.description + "%s" % time)

    def write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def delete():
        sys.stdout.write('\r')
        sys.stdout.flush()


class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "%.3f s" % self.elapsed


class ProgressBar:
    def __init__(self, max_value, start_description, description_max_size, carriage_return):
        self.description_max_size = description_max_size
        self.start_description = start_description
        if carriage_return:
            cr = '\n'
        else:
            cr = ''
        self.widgets = [pb.FormatCustomText('%(label)s%(space)s', dict(label=self.start_description,
                        space=(max(self.description_max_size - len(self.start_description), 1)*" "))),
                        ' ', pb .Percentage(), pb.Bar(), pb.Timer(), cr]
        self.progress_bar = pb.ProgressBar(widgets=self.widgets).start(max_value)

    def update(self, value, description):
        self.widgets[0].update_mapping(label=description,
                                       space=(max(self.description_max_size - len(description), 1) * " "))
        self.progress_bar.update(value)


def countdown(n):
    while n > 0:
        n -= 1


if __name__ == '__main__':
    test = ProgressBar(5, "test", 20, False)
    time.sleep(1)
    test.update_description("salut")
    time.sleep(2)
    test.update_description("Blabla")
    time.sleep(2)

