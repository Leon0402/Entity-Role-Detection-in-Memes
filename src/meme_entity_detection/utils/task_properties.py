label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
num_classes = len(label2id)
labels = [item[0] for item in sorted(label2id.items(), key=lambda item: item[1])]
