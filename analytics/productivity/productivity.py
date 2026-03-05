'''
This is the productivity app where you can track your tasks and get reminders to stay on track.
Each tasks has a duration in minutes, when the python script is executed, it will start the tasks 
with its countdown timer.

For visualizations purposes, the app is run in a separated terminal window.
'''

from asyncio import tasks
import os # gets the file paths.
import json # reads and parse the JSON file.
import time # pausese the program (time.sleep).
from datetime import datetime, timedelta #calculate task start/end times
from termcolor import cprint # prinst colored text in the terminal
import random # chooses random motivational reminders

def load_tasks():
    '''
    Function to import json file with tasks
    '''

    # Finds the folder where the Python script is located.
    current_dir = os.path.dirname(__file__)

    # Builds the full path to tasks.json 
    tasks_path = os.path.join(current_dir, "tasks.json")

    # Opens the JSON file in read mode and assigns it to f.
    # Pythons context manager (with ... as f) assign the opened file to a variable f 
    # instead of 'f' you could use other naming as json_file, etc
    with open(tasks_path, "r") as f:
        tasks = json.load(f)

    return tasks

def open_reminders():
    '''
    Read reminders.txt file and return a list of reminders
    '''
    # Finds the folder where the Python script is located.
    current_dir = os.path.dirname(__file__)

    # Builds the full path to tasks.json 
    reminders_path = os.path.join(current_dir, "reminders.txt")

    with open(reminders_path, "r") as f:
        reminders = [line.strip() for line in f]
    return reminders



def get_task_schedule(tasks):
    # Sets the start time of the first task to the current time.
    task_start_time = datetime.now()

    # Creates an empty list that will store the full schedule.
    schedule = []

    # When Python reads a JSON file, it returns a dictionary
    # Each element in the dictionary is a tuple: (key, value)
    # Python allows tuple unpacking in for loops.
    # On each iteration:
    # - The first element of the tuple (the key) is assigned to task.
    # - The second element of the tuple (the value) is assigned to minutes.
    for task, minutes in tasks.items():

        # Calculates the end time of the task by adding the duration to the start time.
        end_time = task_start_time + timedelta(minutes=minutes)

        # Adds a tuple (task name, start time, end time) to the schedule list.
        schedule.append((task, task_start_time, end_time))

        # Updates the start time for the next task to the current task’s end time.
        task_start_time = end_time 

    return schedule 

def main():
    tasks = load_tasks() # Loads tasks from JSON.
    schedule = get_task_schedule(tasks) #Creates the schedule list.
    current_index = 0 #Keeps track of which task is currently active.

    while True:

        # Current time
        now = datetime.now()

        # Gets the current task using current_index.
        current_task, start_time, end_time = schedule[current_index]
        remaining_time = end_time - now 
        remaining_minutes = int(remaining_time.total_seconds() // 60)

        print('')

        for index, (task, s_time, e_time) in enumerate(schedule):
            if index < current_index:
                # task is completed 
                print(f'{task} done: {e_time.strftime("%H:%M")}')
            elif index == current_index:
                # curent task 
                if remaining_minutes < 2:
                    cprint('{task} < 2m left!', 'white', 'on_red', attrs=['blink'])
                elif remaining_minutes < 5:
                    cprint(f'{task} - {remaining_minutes} mins', 'white', 'on_red')
                else:
                    cprint(f'{task} - {remaining_minutes} mins', 'white', 'on_blue')
            else:
                print(f'{task} @ {s_time.strftime("%H:%M")}')

        # Loads the motivational quotes
        reminders = open_reminders()

        random_reminder = random.choice(reminders)
        print(random_reminder)

        # Checks if current task is finished (now >= end_time).
        if now >= end_time:

            #If finished:Move to the next task (current_index += 1).
            current_index += 1 

            # If there are no more tasks, print a success message and exit the loop.
            if current_index >= len(schedule):
                cprint("All tasks are completed", 'white', 'on_green')
                break 

        #  Sleep for 5 minutes (300 seconds)
        time.sleep(300)

main()
