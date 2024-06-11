#####################################################################################################
#####################################################################################################
#####################################################################################################
####################### Generating XSmall Datasets for Rapid Screening Testing #######################
#######################                Informative Features = 2               #######################
#######################                Correlated Features = 2                #######################
#######################                Uninformative Features = 4            #######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,2,1)
  x2= rnorm(n,5,2)
  noise= rnorm(n,0,2)
  y= x1*5+x2+noise
  df=data.frame(matrix(data=c(x1,x2),
                       nrow=n, 
                       ncol = 2, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:2), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 2+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:2), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 2+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:2), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_DXS",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_DXS",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 1
nDF = 1
for(i in 1:nDF){
  var_name = paste("dxs",i,sep="")
  assign(var_name, datgen(300,3,1,init_seed+i-1,i))
  write.csv(get(var_name), paste("./DXS",i,".csv", sep=""), row.names = FALSE)
}

dxsTEST= datgen(300,3,1,1050,'TEST')
write.csv(dxsTEST, "./DXS_TEST.csv", row.names = FALSE)
  

#####################################################################################################
#####################################################################################################
#####################################################################################################
####################### Generating Small Datasets for Rapid Screening Testing #######################
#######################                Informative Features = 3               #######################
#######################                Correlated Features = 2                #######################
#######################                Uninformative Features = 10            #######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,2,1)
  x2= rnorm(n,5,2)
  x3= rnorm(n,3,3)
  noise= rnorm(n,0,2)
  y= x1*5+x2+x3*2+noise
  df=data.frame(matrix(data=c(x1,x2,x3),
                       nrow=n, 
                       ncol = 3, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:3), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 3+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:3), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 3+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:3), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_DS",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_DS",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 100
nDF = 1
for(i in 1:nDF){
  var_name = paste("ds",i,sep="")
  assign(var_name, datgen(300,10,2,init_seed+i-1,i))
  write.csv(get(var_name), paste("./DS",i,".csv", sep=""), row.names = FALSE)
}

dsTEST= datgen(150,10,2,1050,'TEST')
write.csv(dsTEST, "./DS_TEST.csv", row.names = FALSE)

#####################################################################################################
#####################################################################################################
#####################################################################################################
###################### Generating Medium Datasets for Rapid Screening Testing  ######################
#######################                Informative Features = 6               #######################
#######################                Correlated Features = 4                #######################
#######################                Uninformative Features = 50           #######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,1,1)
  x2= rnorm(n,2,3)
  x3= rnorm(n,5,4)
  x4= rnorm(n,4,1)
  x5= rnorm(n,3,3)
  x6= rnorm(n,2,4)
  noise= rnorm(n,0,2)
  y= x1*2+x2*5+x3/3+x4-x5/4+x6*2+noise
  df=data.frame(matrix(data=c(x1,x2,x3,x4,x5,x6),
                       nrow=n, 
                       ncol = 6, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:6), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 6+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:6), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 6+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:6), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_DM",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_DM",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 1000
nDF = 1
for(i in 1:nDF){
  var_name = paste("dm",i,sep="")
  assign(var_name, datgen(300,50,4,init_seed+i-1,i))
  write.csv(get(var_name), paste("./DM",i,".csv", sep=""), row.names = FALSE)
}

dmTEST= datgen(600,50,4,2050,'TEST')
write.csv(dmTEST, "./DM_TEST.csv", row.names = FALSE)




#####################################################################################################
#####################################################################################################
#####################################################################################################
####################### Generating Large Datasets for Rapid Screening Testing #######################
#######################               Informative Features = 10               #######################
#######################                Correlated Features = 5                #######################
#######################                Uninformative Features = 135           #######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,3,2)
  x2= rnorm(n,5,1)
  x3= rnorm(n,2,3)
  x4= rnorm(n,2,2)
  x5= rnorm(n,4,3)
  x6= rnorm(n,1,3)
  x7= rnorm(n,5,2)
  x8= rnorm(n,4,2)
  x9= rnorm(n,3,3)
  x10= rnorm(n,1,2)
  noise= rnorm(n,0,2)
  y= x1/2+x2+x3*3+x4-x5/5+x6-x7/3+x8*2-x9*3+x10*4+noise
  df=data.frame(matrix(data=c(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10),
                       nrow=n, 
                       ncol = 10, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:10), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:10), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:10), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_DL",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_DL",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 10000
nDF = 1
for(i in 1:nDF){
  var_name = paste("dm",i,sep="")
  assign(var_name, datgen(300,135,5,init_seed+i-1,i))
  write.csv(get(var_name), paste("./DL",i,".csv", sep=""), row.names = FALSE)
}

dlTEST= datgen(1500,135,5,3050,'TEST')
write.csv(dlTEST, "./DL_TEST.csv", row.names = FALSE)



#####################################################################################################
#####################################################################################################
#####################################################################################################
####################### Generating XLarge Datasets for Rapid Screening Testing ######################
#######################               Informative Features = 10                ######################
#######################                Correlated Features = 20                ######################
#######################                Uninformative Features = 220            ######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,3,2)
  x2= rnorm(n,5,1)
  x3= rnorm(n,2,3)
  x4= rnorm(n,2,2)
  x5= rnorm(n,4,3)
  x6= rnorm(n,1,3)
  x7= rnorm(n,5,2)
  x8= rnorm(n,4,2)
  x9= rnorm(n,3,3)
  x10= rnorm(n,4,2)

  noise= rnorm(n,0,2)
  y= x1/2+x2+x3*3+x4-x5/5+x6-x7/3+x8*2-x9*3+x10*4+noise
  df=data.frame(matrix(data=c(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10),
                       nrow=n, 
                       ncol = 10, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:10), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:10), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:10), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_DXL",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_DXL",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 100000
nDF = 1
for(i in 1:nDF){
  var_name = paste("dxl",i,sep="")
  assign(var_name, datgen(300,220,20,init_seed+i-1,i))
  write.csv(get(var_name), paste("./DXL",i,".csv", sep=""), row.names = FALSE)
}

dxlTEST= datgen(2500,220,20,4050,'TEST')
write.csv(dxlTEST, "./DXL_TEST.csv", row.names = FALSE)


#####################################################################################################
#####################################################################################################
#####################################################################################################
####################### Generating 2XL Datasets for Rapid Screening Testing    ######################
#######################               Informative Features = 10                ######################
#######################                Correlated Features = 40                ######################
#######################                Uninformative Features = 400            ######################
#####################################################################################################
#####################################################################################################
#####################################################################################################


############################
# data generating function
datgen=function(n,uninf,corr,seed,ITR){
  ########
  # This function saves the list of 
  # coefficients and powers of correlated variables 
  # in the current folder.
  ########
  set.seed(seed)
  x1= rnorm(n,3,2)
  x2= rnorm(n,5,1)
  x3= rnorm(n,2,3)
  x4= rnorm(n,2,2)
  x5= rnorm(n,4,3)
  x6= rnorm(n,1,3)
  x7= rnorm(n,5,2)
  x8= rnorm(n,4,2)
  x9= rnorm(n,3,3)
  x10= rnorm(n,4,2)
  noise= rnorm(n,0,2)
  y= x1/2+x2+x3*3+x4-x5/5+x6-x7/3+x8*2-x9*3+x10*4+noise
  df=data.frame(matrix(data=c(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10),
                       nrow=n, 
                       ncol = 10, 
                       byrow = FALSE,
                       dimnames = list(c(1:n),c(paste0("x",c(1:10), sep="")))))
  coef.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr,
                            dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                            c(paste0("coef.x",c(1:10), sep=""),paste0("coef.corr",c(1:corr),sep="")))))
  exp.df=data.frame(matrix(NA, nrow = corr, ncol = 10+corr, 
                           dimnames = list(c(paste0("corr",c(1:corr),sep="")),
                                           c(paste0("exp.x",c(1:10), sep=""),paste0("exp.corr",c(1:corr),sep="")))))
  for(i in 1:corr){
    csel=sample(1:ncol(df),2,replace = F)
    coef1=sample(-10:10,1)
    exp1= sample(-2:2,1)
    coef.df[i,unique(csel)]=coef1
    exp.df[i,unique(csel)]=exp1
    newvar= coef1*(rowSums(df[,csel]))^(exp1)+rnorm(n)
    df=as.data.frame(cbind(df,newvar))
    colnames(df)[ncol(df)]=paste0("corr",i)
  }
  for(i in 1:uninf){
    df=as.data.frame(cbind(df,rnorm(n, runif(1,1,5), runif(1,1:3))))
    colnames(df)[ncol(df)]=paste0("noise",i)
  }
  write.csv(coef.df, paste("./CorrData/coefcorr_D2XL",ITR,".csv",sep=""))
  write.csv(exp.df, paste("./CorrData/expcorr_D2XL",ITR,".csv",sep=""))
  return(as.data.frame(cbind(df,y)))
}

#### Experiments #####
init_seed = 1000000
nDF = 1
for(i in 1:nDF){
  var_name = paste("d2xl",i,sep="")
  assign(var_name, datgen(300,400,40,init_seed+i-1,i))
  write.csv(get(var_name), paste("./D2XL",i,".csv", sep=""), row.names = FALSE)
}

d2xlTEST= datgen(4500,400,40,5050,'TEST')
write.csv(d2xlTEST, "./D2XL_TEST.csv", row.names = FALSE)


#####################################################################################################
#                                                TESTING
#####################################################################################################
