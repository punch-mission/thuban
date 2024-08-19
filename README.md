# thuban

High precision astrometric image solving from a guess.

> [!WARNING]
> This package is not yet ready for production usage.


## To-do
 - [x] make a basic pointing refinement algorithm
 - [x] make a CLI for distortion and pointing solving simultaneously
 - [ ] make a distortion only solving step
 - [ ] try the distortion solving with a different size to see if that improves the pointing
 - [ ] make a pointing report that includes all the helpful info at solving results
 - [ ] evaluate quality of fit and report
 - [ ] move steps from the CLI to a function for cleaner code
 - [ ] allow passing in or using the existing distortion model instead of blanking it everytime
 - [ ] check that all images have the same format upon loading
 - [ ] avoid loading all the images at the beginning to preserve RAM
 - [ ] allow solving by passing in a single filename instead of just a directory
 - [ ] add more tests
 - [ ] document all functions
 - [ ] make a documentation website
