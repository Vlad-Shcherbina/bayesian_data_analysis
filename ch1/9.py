from __future__ import division
import logging
import random

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

# everything is in minutes

AVERAGE_PATIENT_FREQUENCY = 1.0 / 10
MIN_APPOINTMENT_TIME = 5
MAX_APPOINTMENT_TIME = 20
NUM_DOCTORS = 3

# event types
NEW_PATIENT = 'NEW_PATIENT'
APPOINTMENT_END = 'APPOINTMENT_END_'  # plus doctor number

class Simulator(object):
    def __init__(self, start_time, end_time, prng=None):
        if prng is None:
            prng = random
        self.prng = prng

        self.time = start_time
        self.end_time = end_time

        self.event_queue = []
        logger.info('simulation begins')

        self.schedule_next_patient()

        self.doctor_free = [True] * NUM_DOCTORS
        self.queue = []  # arrival times of waiting patients

        self.num_patients = 0
        self.num_waiting_patients = 0
        self.total_wait_time = 0

    def schedule_next_patient(self):
        t = self.prng.expovariate(1.0 / 10)
        if self.time + t <= self.end_time:
            self.event_queue.append((self.time + t, NEW_PATIENT))

    def schedule_appointment_end(self, doctor_index):
        assert self.doctor_free[doctor_index]
        self.doctor_free[doctor_index] = False
        t = self.prng.uniform(MIN_APPOINTMENT_TIME, MAX_APPOINTMENT_TIME)
        self.event_queue.append(
            (self.time + t, APPOINTMENT_END + str(doctor_index)))

    def handle_event(self, event):
        time, type = event
        logger.info('{:.2f}m passed...'.format(time - self.time))
        self.time = time

        if type == NEW_PATIENT:
            self.num_patients += 1
            self.schedule_next_patient()
            logger.info('new patient has arrived')
            if any(self.doctor_free):
                self.schedule_appointment_end(self.doctor_free.index(True))
            else:
                self.queue.append(self.time)
        elif type.startswith(APPOINTMENT_END):
            index = int(type[len(APPOINTMENT_END):])
            assert not self.doctor_free[index]
            logger.info('doctor {} ended his appointment'.format(index))
            self.doctor_free[index] = True
            if self.queue:
                logger.info('and took waiting patient')
                wait_time = self.time - self.queue.pop(0)
                self.total_wait_time += wait_time
                self.num_waiting_patients += 1
                self.schedule_appointment_end(index)
            else:
                logger.info('and has nothing to do')
        else:
            assert False, type

    def run(self):
        while self.event_queue:
            logger.info('-' * 20)
            logger.info('Time: {:.2f}'.format(self.time))
            logger.info('Event queue: {}'.format(self.event_queue))
            logger.info('Patient queue: {}'.format(self.queue))
            logger.info('Doctors free: {}'.format(self.doctor_free))
            event = min(self.event_queue)
            self.event_queue.remove(event)
            self.handle_event(event)


def main():
    #logging.basicConfig(level=logging.INFO)

    s = Simulator(0, 7 * 60, prng=random.Random(42))
    s.run()

    print s.num_patients, 'patients'
    print s.num_waiting_patients, 'patients that had to wait for a doctor'
    if s.num_waiting_patients:
        print 'their average wait: {:.2f}m'.format(
            s.total_wait_time / s.num_waiting_patients)
    print 'office closed {:.2f}m after it stopped admitting patients'.format(
        max(0, s.time - s.end_time))

    # Run multiple simulations.

    num_patients_distribution = []
    num_waiting_patients_distribution = []
    average_wait_distribution = []
    close_time_distribution = []
    for i in range(1000):
        s = Simulator(0, 7 * 60, prng=random.Random(42 + i))
        s.run()
        num_patients_distribution.append(s.num_patients)
        num_waiting_patients_distribution.append(s.num_waiting_patients)
        if s.num_waiting_patients:
            average_wait_distribution.append(
                s.total_wait_time / s.num_waiting_patients)
        else:
            average_wait_distribution.append(0)
        close_time_distribution.append(max(0, s.time - s.end_time))

    plt.subplot(221)
    plt.title('number of patients')
    plot_distribution(num_patients_distribution)

    plt.subplot(222)
    plt.title('number of waiting patients')
    plot_distribution(num_waiting_patients_distribution)

    plt.subplot(223)
    plt.title('average wait time')
    plot_distribution(average_wait_distribution)

    plt.subplot(224)
    plt.title('close time')
    plot_distribution(close_time_distribution)

    plt.show()


def plot_distribution(distribution):
    distribution.sort()
    plt.hist(distribution)
    plot_height = plt.ylim()[1]
    n = len(distribution)
    plt.plot(
        [distribution[i * n // 4] for i in range(1, 4)],
        [0.99 * plot_height] * 3,
        'ko-')

    # Find shortest 50% interval.
    intervals = zip(distribution[:n//2], distribution[n//2:])
    shortest = min(intervals, key=lambda(a, b): b - a)
    plt.plot(
        shortest,
        [0.95 * plot_height] * 2,
        'go-')


if __name__ == '__main__':
    main()
