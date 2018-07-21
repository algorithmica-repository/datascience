from sklearn import model_selection

X = list(range(0, 120, 10))
y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
skf = model_selection.StratifiedKFold(n_splits=3)
splits = skf.split(X, y)

for train, test in splits:
    print("%s %s" % (train, test))
    
srh = model_selection.StratifiedShuffleSplit(3, test_size=0.2)
splits = srh.split(X, y)
for train, test in splits:
    print("%s %s" % (train, test))