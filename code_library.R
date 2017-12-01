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

apply_limma <- function(dataframe, design_dataframe, adjust="BH") {

	# Load packages
	require(limma)
	require(edgeR)

	# Create design matrix
	design.mat <- as.matrix(design_dataframe)

	# Create DGEList object
    dge <- DGEList(counts=dataframe)

    # Calculate normalization factors
    dge <- calcNormFactors(dge)

    # Run VOOM
    v <- voom(dge, plot=TRUE)

	# Fit linear model
	fit <- lmFit(v, design.mat)

	# Create contrast matrix
	contrast.mat <- makeContrasts(Diff = samples - controls, levels = design.mat)

	# Fit
	fit2 <- contrasts.fit(fit, contrast.mat)

	# Run DE
	fit3 <- eBayes(fit2)

	# Get results
	deg <- topTable(fit3, adjust=adjust, number=nrow(dataframe))
 
	# Return
	return(deg)
}
