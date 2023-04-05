# Hypothesis 
Precomputing a subset data can greatly improve on the speed of the video queries without sacrificing a (substantial) amount of accuracy.

# Link between the user perspective and the technical solution
The biggest convenience of Shazam is the way it can quickly find the source of the audio it is presented no matter at which you point the user starts recording. As a user of Video Shazam, I want to have that speed for the pipeline to be usable. An issue with the video query system from Lab 5 was how slow the feature matching was for an entire video (due to the bruteforce method it used). The team wants to maximize the speedup the pipeline by pre-computing both single- and multimodal signatures and minimize the corresponding decrease in accuracy of the final product.

# Motivation of the proposed solution (why is it, in theory, better than alternatives?)
A computation over every frame of every video in the database is excessive. The proposed method would have the most intensive database computations done in advance. The idea is to have the comparison be done through a signature for the entire video as well as multiple small segments over the entire video, instead of through a frame-by-frame sliding window. In theory, this solution should save the majority of the computation and make the Video Shazam experience more seamless. All that would be left is the the computation of the desired feature/s on the input video and the comparison with each precomputed signature.

# Innovation beyond the material learned in the labs
An issue the team realized would come up is that of the differencs between segments in the same video. While on average it is fair to say that a single video looks a certain way, there are sometimes small clips/segments of it that may look radically different (e.g. a scene change in a movie, a cut in a news report, etc.). Thus, computing a single signature per video will likely not be sufficient.

The solution the team came up with was segmenting every video in the database into 5-second chunks. The pipeline will use those for the identification of the video segment provided as well. That way even when a user gives an arbitrary and seemingly unique chunk of a video, the pipeline has a chance of finding it *hopefully* faster than the brute-force method from the labs.

Another issue that is currently expected to come up is that of the user inputing a video segment that is between two or more of the 5-second chunks in our database (e.g. 00:03 - 00:08 or 00:03 - 00:17). A possible solution is computing signatures for each pairr of 5-second segments as well. However, that part has not been implemented yet.

# Quantitative evaluation and qualitative analysis of when it works and doesn't work and why
The team will focus on three criterion for evaluating our system. Those will be:

- Correct video identification: Did the pipeline manage to find the correct video at all?
- Correct segment identification: Did the pipeline manage to find exactly which part of the videos in the database it was given?
- Time taken: Did the signature method manage to output an answer any faster than the brute-force sliding window method?

Based on the combination of these three factors we will consider whether our system is a success or not. As far as the results we are expecting, we expect better time performance with a decrease in accuracy (due to the nature of limiting the comparisons possible). If the decrease in accuracy is substantial enough, even if the system is faster, it would be deemed inefficient. A *substantial enough* decrease we define as it failing to identify a large amount of clear segment

# Pitch presented during the Lab

# Assumptions made for the assignment
- (At least so far) Perfect Localization (still video in video, but well-localized)
- The average user video segment given will not be more than 15 seconds (from a user perspective that would also likely hold up)
- All videos will have been recorded with the same mobile phone camera (and by extension microphone)

# Invariances of the system
- The pipeline aims to be *scene and shot invariant*. It should not matter which scene/shot (or even overlap of the two) from a video a user provides, the system should still be able to find it
- The pipeline should be able to work for any of the features it is performed on (color histograms, mfccs, some multimodal criterion, yada yada yada)
