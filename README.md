SoundScore is an attempt to simplify the testing of audio components for noise and distortion. By comparing a known source signal to one played back off your hardware the program will generate a "Sound Score" out of 100 to give you a rough non-scientific benchmark of how your hardware compares. 

At its simplest you are playing a clean .wav file, either from your computer or via the source you want to test like a CDJ or a mp3 player on your phone, that signal is connected to the line in on your computer or audio interface. This allows the SoundScore program to create a baseline. You can then change out components or cable to see how they compare to the baseline. or just see how your gear stacks up in general.  

To install:
Unzip all files into the same folder, open a terminal from within that folder and run: 

pip install -r requirements.txt 
python main.py
 


