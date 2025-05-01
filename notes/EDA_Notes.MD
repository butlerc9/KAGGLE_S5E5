
# Things to check
- Does episode number increase with id, if so then we know the dataset is ordered and we can order episdes without a name
- 

# Feature engineering
- Drop podcast name
- Parse out the number of the episode and convert it to an int
- Some episode lengths are missing, these could be imputed. It must be at least as long as the maximum listening time
- Drop genre for now
- Create a flag for weekend or not weekend
- Drop time of day
- Does the host popularity increase
- Creat a flag for ads or no ads

# Models
1. Modal value
2. Average podcast value