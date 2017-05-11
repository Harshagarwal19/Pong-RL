# Abstract
Pong is one of the most popular games ever developed in the world. We have all played it at least once in our lifetime, in some platform - maybe on an arcade ma- chine, a hand-held device or on the web. It is a simple game with little to no learning curve. However, as triv- ial as the task may seem to humans, it was difficult for a computer to learn to play it just by looking at the game - that is, learning by only analyzing the image of the frames, without any explicit knowledge or additional in- formation about the game environment. In 2013, Deep- Mind Technologies published a seminal paper “Playing Atari with Deep Reinforcement Learning” (Mnih et al. 2013)[page - 1] that did just that. In this paper, Mnih et al. employed deep neural networks, model-free reinforcement learning and various implementation techniques to successfully teach a computer to play seven different Atari games, including Pong, with a single algorithm and only game screen images as input. This project aims to understand their work and try to replicate the results on Pong.


# Environment
[Open AI gym (beta)]

# Language and Libraries
* python 2.7
* Tensorflow
* numpy
* matplotlib
* pickle
* scipy
* Python Imaging Library (PIL)

# Instructions
* After installing the necessary libraries, run 
``
python DriverProgram.py
``
* Training will start with the saved CNN weights in networks folder. To start from scratch, delete the contents of network folder. 

# Additional Resources
+ For introduction to model-free reinforcement learning and Mnih et al. paper - [projectPaper] 
+ Presentation - [link]
+ A video of the code in action - [YouTube link]

   [Open AI gym (beta)]: <https://gym.openai.com/>
   [projectPaper]: <https://drive.google.com/file/d/0BygLf1QZV3ixVWROOHAtenpXRU0/view?usp=sharing>
   [YouTube link]: <https://www.youtube.com/watch?v=DDsVLMTTdZ4>
   [link]: <https://docs.google.com/presentation/d/1cu-CqZ7BLPQLXPjuXxekJK486gOdK5lk9UcuU7HlG9k/edit?usp=sharing>
