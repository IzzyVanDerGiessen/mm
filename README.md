## Setting up the database
All of the database videos (in the '/videos' folder) and signatures should already be in the actual repository. In the case there are issues with the database, one can simply run the `Database.py` script, ensuring `refresh_database = True`.

```css
py Database.py
```
From then on, the queries should run fine.

### How to add new input videos 
In the case of wanting to add more videos, a user would need to add them in the '/videos' folder. Subsequently, they would need to run the 'Cropper.py' script to export the audio data into a '.wav' file. This also goes for creating signatures of different lengths per video.
#### Extracting a .wav file 
```css
py cropper.py --input=True --input_path= 'some/path' 
``` 
#### Cropping video into chunks of length k
```css
py cropper.py --video=True --cliplength=k
```
#### Cropping audio into chunks of length k
```css
py cropper.py --audio=True --cliplength=k
```
## Running a Query
To run an actual query on the Video Shazam, one can either take one of the provided input videos or give one of their own in the manner mentioned above. From then on, there are two pipelines on which the input can be tested, being the brute force pipeline which computes a sliding window over each video in the database. There is also the signature pipeline which uses pre-computed data from the database to estimate the input query's identity faster, but with lower accuracy.

The user can pick any combination of the features `colorhists, mfccs, audio_powers` and `temporal_diff`

### Running the brute-force pipeline
```css
py VideoQuery.py --filepath='some\path\to\video' --pipeline=brute_force -features= mfccs colorhists
```
### Running the signatures pipeline
```css
py VideoQuery.py --filepath='some\path\to\video' --pipeline=signatures -features= mfccs colorhists
```



