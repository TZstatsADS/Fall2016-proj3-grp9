# 0. Package Setup ##################################################################
library(dplyr) 

# 1. Create Image Directories #################################################################################
# 1.1 Create a name list for fried chicken 
dat.chickname <- data.frame()
for(i in 1:1000) {
  # Change to your local path when creating the name list 
  dat.chickname[i, 1] <- paste0(sprintf("/Users/yanjin1993/GitHub/Fall2016-proj3-grp9/data/images/chicken_%04d", i),".jpg" )  
}

# 1.2 Create a name list for poodle dogs 
dat.dogname <- data.frame()
for(i in 1:1000) {
  # Change to your local path when creating the name list 
  dat.dogname[i, 1] <- paste0(sprintf("/Users/yanjin1993/GitHub/Fall2016-proj3-grp9/data/images/dog_%04d", i),".jpg" )  
}

# Row-bind two name lists together 
dat.namelist <- rbind(dat.chickname, dat.dogname) %>% 
  rename(img_directory = V1)

write.csv(dat.namelist, "/Users/yanjin1993/Google Drive/Columbia University /2016 Fall /Applied Data Science /Project_003/exported_data/dat_namelist.csv")
