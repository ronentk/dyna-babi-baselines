import data_utils
import os

data_dir = "data/tasks_1-20_v1-2/en-10k"

def get_average_story_length(data):
    story_length = 0
    for story, query, answer in data:
        story_length += len(story)
    average_story_length = story_length / len(data)
    return average_story_length


def get_stats(tasks, groups, test=False):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_files = []
    test_files = []
    for task in tasks:
        s = "qa{}_".format(task)
        train_files.extend([(task, f) for f in files if s in f and 'train' in f])
        test_files.extend([(task, f) for f in files if s in f and 'test' in f])

    files = test_files if test else train_files
    for file_tuple in files:
        with open(file_tuple[1]) as file:
            lines = file.readlines()
            n_lines = len(lines)
            group_count = {}
            for group in groups:
                group_count[group[0]] = 0
            for line in lines:
                for group in groups:
                    for element in group[1:]:
                        if element in line:
                            group_count[group[0]] += 1

            for group in groups:
                print("Task {0}:elements from \"{1}\" appear in {2} lines, which are {3:.2f}% of all lines in task {0}".format(file_tuple[0], group[0], group_count[group[0]], ((group_count[group[0]] / n_lines) * 100)))


def get_starting_location(tasks, people, locations):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_files = []
    test_files = []
    for task in tasks:
        s = "qa{}_".format(task)
        train_files.extend([(task, f) for f in files if s in f and 'train' in f])
        test_files.extend([(task, f) for f in files if s in f and 'test' in f])

    for file_tuple in train_files:
        with open(file_tuple[1]) as file:
            lines = file.readlines()

            people_location_counters = []
            for person in people:
                people_location_counters.append({})
                for location in locations:
                    people_location_counters[-1][location] = 0

            counted = [False] * len(people)
            for line in lines:
                if "Where" in line:
                    continue
                if "1" in line.split()[0]:
                    for i in range(len(counted)):
                        counted[i] = False
                for i in range(len(people)):
                    if counted[i] is False:
                        if people[i] in line:
                            for location in locations:
                                if location in line:
                                    counted[i] = True
                                    people_location_counters[i][location] += 1

            for i in range(len(people)):
                print("Task {}: {} has the following location count: {}".format(file_tuple[0], people[i], people_location_counters[i]))

def solve_babi(tasks, people, locations, objects, moves, grabs, drops, test=False):
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_files = []
    test_files = []
    for task in tasks:
        s = "qa{}_".format(task)
        train_files.extend([(task, f) for f in files if s in f and 'train' in f])
        test_files.extend([(task, f) for f in files if s in f and 'test' in f])

    files = test_files if test else train_files
    for file_tuple in files:
        with open(file_tuple[1]) as file:
            lines = file.readlines()

            who_where = [None] * len(people)
            what_where = [None] * len(objects)
            who_got = [None] * len(objects)

            correct = 0
            wrong = 0
            for k, line in enumerate(lines):
                if "Where" in line:
                    for i in range(len(objects)):
                        if objects[i] in line:
                            for location in locations:
                                if location in line:
                                    if location == what_where[i]:
                                        correct += 1
                                    else:
                                        if what_where[i] is not None:
                                            print("Mismatch in line {}: predicted {}, but correct answere was {}".format(k + 1, what_where[i], location))
                                        wrong += 1
                if "1" == line.split()[0]:
                    for i in range(len(people)):
                        who_where[i] = None
                    for i in range(len(objects)):
                        what_where[i] = None
                        who_got[i] = None
                for move in moves:
                    if move in line:
                        for i in range(len(people)):
                            if people[i] in line:
                                for location in locations:
                                    if location in line:
                                        who_where[i] = location
                                        for j in range(len(objects)):
                                            if who_got[j] == people[i]:
                                                what_where[j] = location
                for grab in grabs:
                    if grab in line:
                        for i in range(len(people)):
                            if people[i] in line:
                                for j in range(len(objects)):
                                    if objects[j] in line:
                                        who_got[j] = people[i]
                                        if who_where[i] is not None:
                                            what_where[j] = who_where[i]
                for drop in drops:
                    if drop in line:
                        for i in range(len(people)):
                            if people[i] in line:
                                for j in range(len(objects)):
                                    if objects[j] in line:
                                        who_got[j] = None
                                        if who_where[i] is not None:
                                            what_where[j] = who_where[i]
        print("correct = {}".format(correct))
        print("wrong = {}".format(wrong))



# for task in [2, 11, 21, 22]:
#     task_train, task_test = data_utils.load_task(data_dir, task)
#     print(task)
#     print(get_average_story_length(task_train + task_test))

get_stats([2, 22], [["Moves", "moved", "went", "journeyed", "travelled"], ["Grabs", "grabbed", "took", "got", "picked up"], ["Drops", "dropped", "discarded", "put down", "left"], ["Where", "Where"]])
# get_stats([2, 22], [["John", "John"], ["Daniel", "Daniel"], ["Mary", "Mary"], ["Sandra", "Sandra"]])
# get_stats([2, 22], [["bedroom", "bedroom"], ["bathroom", "bathroom"], ["garden", "garden"], ["kitchen", "kitchen"], ["office", "office"], ["hallway", "hallway"]])

# get_starting_location([2, 22], ["John", "Daniel", "Mary", "Sandra"], ["bedroom", "bathroom", "garden", "kitchen", "office", "hallway"])
# solve_babi([2, 22],
#            ["John", "Daniel", "Mary", "Sandra"],
#            ["bedroom", "bathroom", "garden", "kitchen", "office", "hallway"],
#            ["football", "apple", "milk"],
#            ["moved", "went", "journeyed", "travelled"],
#            ["grabbed", "took", "got", "picked up"],
#            ["dropped", "discarded", "put down", "left"])