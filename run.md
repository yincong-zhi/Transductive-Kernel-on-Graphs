# Commands for Real World Data Results

## Texas
```
TGGP
python main.py --data=Texas --poly -1 1 0 0 0
81.1%

training on validation
python main.py --data=Texas --train_on_val=True --poly 0 0.5 0 0 0
86.5%
```

## Wisconsin
```
TGGP
python main.py --data=Wisconsin --poly 0 0.1 0 0 0
82.4%

training on validation
python main.py --data=Wisconsin --train_on_val=True --poly 0 0 0 0.35 0
86.3%
```

## Cornell
```
TGGP
python main.py --data=Cornell --poly 0 1 0 0 0
75.7%

training on validation
python main.py --data=Cornell --train_on_val=True --poly 0 0.1 0 0 0
81.1%
```

## Chameleon
```
TGGP
python main.py --data=Chameleon --poly 0 0.5 0 0 0
63.2%

training on validation
python main.py --data=Chameleon --train_on_val=True --poly 0 0.4 0 0 0
63.4%
```

## Cora
```
TGGP
python main.py --epoch=31 --poly 0 0.5 0 0 0
80.3%

training on validation
python main.py --train_on_val=True --epoch=31 --poly 0 2.5 0 0 0
83.8%
```

## Citeseer
```
TGGP
python main.py --data=Citeseer --poly 0 22 0 0 0
70.5%

training on validation
python main.py --data=Citeseer --train_on_validation=True --poly 0 22 0 0 0
76.7%
```

## Squirrel
```
TGGP
python main.py --data=Squirrel --poly 0 0.1 0 0 0
53.8%

train on validation
python main.py --data=Squirrel --train_on_val=True --poly 0 0.5 0 0 0
54.2%
```

## Actor
```
TGGP
python main.py --data=Actor --poly 0 0.1 0 0 0
34.9

train on validation
python main.py --data=Actor --train_on_val=True --poly 0 1 0 0 0
36.9
```