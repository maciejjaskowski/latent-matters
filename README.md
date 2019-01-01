

# Latent space matters

This is an ongoing research


# Install
Install dependencies (see Dockerfile for reference)

```
git submodule init
git submodule update
pip install -r requirements.txt
```


# Run

## Generate observations

This will generate 100k observations of PongNoFrameskip-v4
```
python generate_observations.py --env PongNoFrameskip-v4 --destination-dir "../observations-PongNoFrameskip-v4"
```