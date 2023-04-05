
# Hypothesis {#hypothesis .unnumbered}

Precomputing a subset data can greatly improve on the speed of the video
queries without sacrificing a (substantial) amount of accuracy.

# Link between the user perspective and the technical solution {#link-between-the-user-perspective-and-the-technical-solution .unnumbered}

The biggest convenience of Shazam is the way it can quickly find the
source of the audio it is presented no matter at which you point the
user starts recording. As a user of Video Shazam, I want to have my
queries return an output (even one that is slightly off) in a reasonable
time. An issue with the video query system from Lab 5 was how slow the
feature matching was for an entire video (due to the sliding window
method it used). The team wants to maximize the speedup of the pipeline
by pre-computing both single- and multimodal signatures and minimize the
corresponding decrease in accuracy of the final product.

# Motivation of the proposed solution (why is it, in theory, better than alternatives?) {#motivation-of-the-proposed-solution-why-is-it-in-theory-better-than-alternatives .unnumbered}

A computation with a sliding window over every frame of every video in
the database is excessive. The proposed method would have the pipeline's
most intensive computations done in advance. The idea is to have the
comparison be done through a signature on the entire video as well as
multiple small segments over the entire video, instead of through a
frame-by-frame sliding window. In theory, this solution should save the
majority of the computation and make the Video Shazam experience more
seamless. All that would be left is the the computation of the desired
feature/s on the input video and the comparison with each precomputed
signature.

# Innovation beyond the material learned in the labs {#innovation-beyond-the-material-learned-in-the-labs .unnumbered}

The way the team came at this innovation was the frustration with the
speed of the queries in Lab 5. Waiting more than a minute (in some cases
even more) made the system practically unusable in a non-lab scenario.
This is what led to the realization that most of those computations are
redundant and the idea of having pre-computed signatures. From then on,
the team was focused on solving the issues from that approach.\
The first issue the came up was that of the differences between segments
in the same video. While, on average, it is fair to say that a single
video looks a certain way, there are sometimes small clips/segments of
it that may look radically different (e.g. a scene change in a movie, a
cut in a news report, etc.). If a user inputs one of those sections,
having a single signature for the entire video will likely not be
sufficient. The solution the team came up with was segmenting every
video in the database into 5-second chunks. The pipeline will use those
for the identification of the video segment provided as well. That way
even when a user gives an arbitrary and seemingly unique chunk of a
video, the pipeline has a chance of finding it *hopefully* faster than
the brute-force method from the labs.\
Another issue that the team expects to come up is that of the user
inputting a video segment that is between two or more of the 5-second
chunks in our database (e.g. 00:03 - 00:08 or 00:03 - 00:17). A possible
solution is computing signatures for each pair of 5-second segments as
well. This can be extended for queries spanning multiple 5-second chunks
by having 10-,20-,30-second chunks, and more. However, that part has not
been implemented yet due to time worries about the database size, so it
has been left as a future improvement.

# Quantitative evaluation and qualitative analysis of when it works and doesn't work and why. {#quantitative-evaluation-and-qualitative-analysis-of-when-it-works-and-doesnt-work-and-why. .unnumbered}

The team will focus on three criterion for evaluating our system. Those
will be:

-   **Correct video identification**: Did the pipeline manage to find
    the correct video at all?

-   **Correct segment identification**: Did the pipeline manage to find
    exactly which part of the videos in the database it was given?

-   **Time taken**: Did the signature method manage to output an answer
    any faster than the brute-force sliding window method?

Based on the combination of these three factors we will consider whether
our system is a success or not. As far as the results we are expecting,
we expect faster performance with a decrease in accuracy (due to the
nature of limiting the comparisons possible). If the decrease in
accuracy is substantial enough, even if the system is faster, it would
be deemed inefficient. A *substantial enough* decrease we define as it
failing to identify a large amount of clear segment. This issue can be
fixed by adding more data.

# Pitch presented during the Lab {#pitch-presented-during-the-lab .unnumbered}

As was clear during the labs, a brute force way of comparing videos
frame by frame is too slow for a usable Shazam-like system. This is why
the team decided to innovate on the way videos are compared by creating
a signature for every full video and every five second segment. The
first step in the pipeline, after pre-processing, is to compute a
signature for the query video. Instead of comparing every frame in video
segments, only the signature is compared and the k best results are then
used to find the best match of the query video. Our goal is to improve
performance without harming the accuracy too much. This is why time is
our main evaluation metric, which we examine quantitatively. A shorter
run time without a noticeable decrease in accuracy is considered a
success. In the future, a tree-like structure could be created where the
bottom layer is five seconds, the next layer ten seconds, then twenty
and so on. This way the system would be better at dealing with long
videos. For now, we assume the query videos are on average about 15
seconds long.

# Assumptions made {#assumptions-made .unnumbered}

-   **Perfect Localization** (still video in video, but well-localized).
    The innovation wants to abstract from the variability of
    localization to test the speed improvement as much as possible.

-   **The average user video segment given will not be more than 15
    seconds**. This is both to mimic a realistic user scenario and to
    minimize the amount of data needed for the experiment. **All input
    videos will have been recorded with the same camera** (and by
    extension microphone) or be of the computer screen recordings. This
    is to minimize the variability of the inputs (except the segments
    they give, of course).

# Invariances of the system {#invariances-of-the-system .unnumbered}

-   The pipeline aims to be \*scene and shot invariant\*. It should not
    matter which scene/shot (or even overlap of the two) from a video a
    user provides, the system should still be able to find it

-   The pipeline should be able to work for any of the features it is
    performed on (color histograms, mfccs, some multimodal criterion,
    yada yada yada)

# Analysis of the code {#analysis-of-the-code .unnumbered}

The code has been organized in different files with specific purposes:\
\
The `Database.py` file creates and loads the database, which consists of
full and cropped videos.\
\
The `Signatures.py` file contains all the methods for computing
signatures, which, as explained above, play an important role in our
retrieval system. Currently, the system has the capacity to create
signatures for four basic features (taken from the lab):
`colorhists, mfccs, audio_powers, temporal_difference`. The system can
be extended for an $n$ amount of features, including multi modal ones.\
\
To run the actual pipeline, `VideoQuery.py` is used. This is the most
important part of our code, since it has methods for the common brute
force pipeline and our experimental pipelines using signatures. Both of
these are timed while performing the experiment.

## Miscellaneous {#miscellaneous .unnumbered}

-   The team has decided to use the Librosa library for the MFCC
    computation and Scikit for the reading of `.wav` files. As this is
    not the point of the innovation of the project, it should not affect
    result

# Evaluation Results {#evaluation-results .unnumbered}

As the signature functions have not been fully completed, there are no
evaluation results yet.
