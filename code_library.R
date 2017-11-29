##############################################################
##############################################################
############ Parkinson's Disease Analysis Support ############
##############################################################
##############################################################
###################### Author: Lily Sarrafha
###################### Affiliation: Ma'ayan Laboratory
###################### Icahn School of Medicine at Mount Sinai

##############################################################
############# 1 R function for limma (Lily)
##############################################################

apply_limma <- function(dataframe, design_dataframe) {

	# Load packages
	require(limma)
	require(edgeR)

	# Create design matrix
	design.mat <- as.matrix(design_dataframe)

	# Create contrast matrix
	contrast.mat <- makeContrasts(Diff = samples - controls, levels = design.mat)

	# Fit linear model
	fit <- lmFit(dataframe, design.mat)

	# Fit
	fit2 <- contrasts.fit(fit, contrast.mat)

	# Run DE
	fit3 <- eBayes(fit2)

	# Get results
	deg <- topTable(fit3, number=nrow(dataframe))
 
	# Return
	return(deg)
}