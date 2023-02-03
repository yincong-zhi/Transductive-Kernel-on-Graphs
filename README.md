# Results on Real World Datasets

## Cora
```
80.3%
python main.py --scipy_max=31 --poly 0 0.5 0 0 0

training on validation
83.8%
python main.py --train_on_val=True --scipy_max=31 --poly 0 2.5 0 0 0
```
<!--
To try
```
python main.py --poly 0 3.5 0 0 0
```
-->
## Citeseer
```
70.5%
python main.py --data=Citeseer --poly 0 22 0 0 0

training on validation
76.7%
python main.py --data=Citeseer --train_on_validation=True --poly 0 22 0 0 0
```
<!--
To try
```
python main.py --data=Citeseer --poly 0 22 0 0 0
```
-->
## Texas
```
81.1%
python main.py --data=Texas --poly -1 1 0 0 0

training on validation
86.5
python main.py --data=Texas --train_on_val=True --poly 0 0.5 0 0 0
```

## Wisconsin
```
82.4%
python main.py --data=Wisconsin --poly 0 0.1 0 0 0

training on validation
86.3%
python main.py --data=Wisconsin --train_on_val=True --poly 0 0 0 0.35 0
```

## Cornell
```
75.7%
python main.py --data=Cornell --poly 0 1 0 0 0

training on validation
81.1%
python main.py --data=Cornell --train_on_val=True --poly 0 0.1 0 0 0
```

## Chameleon
```
63.2%
python main.py --data=Chameleon --poly 0 0.5 0 0 0

training on validation
63.4%
python main.py --data=Chameleon --train_on_val=True --poly 0 0.4 0 0 0
```

## Squirrel
```
53.8%
python main.py --data=Squirrel --poly 0 0.1 0 0 0

train on validation
54.2%
python main.py --data=Squirrel --train_on_val=True --poly 0 0.5 0 0 0
```

## Actor
```
34.9
python main.py --data=Actor --poly 0 0.1 0 0 0

train on validation
36.9
python main.py --data=Actor --train_on_val=True --poly 0 1 0 0 0
```