# centrovision-api

## Setting up project

### Create environment

``` 
python -m venv centrovision-api
```
OR
```
conda create -name centrovision-api
```

### Activate environment

```
source centrovision-api/bin/activate
```
OR

```
conda activate centrovision-api
```

### Install requirement

```
pip install -r requirements.txt
```

### Download model file and place in model/

```
https://drive.google.com/file/d/1NOAYbBDck924rKO0yYP2SNH8nuRJCs5K/view?usp=sharing
```

### Start local server

```
uvicorn main:app --reload
```

### Start remote server

```
USE_NGROK=true uvicorn main:app --reload
```



### API local server running at

```
http://127.0.0.1:8000/docs
```

### API remote inspecter server running at

```
http://127.0.0.1:4040/
```
