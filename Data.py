import os
relative_path = 'data/IRIS.csv'
absolute_path = os.path.dirname(__file__)
full_path = os.path.join(absolute_path, relative_path)
