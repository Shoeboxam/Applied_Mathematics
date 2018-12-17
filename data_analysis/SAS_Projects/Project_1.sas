%LET seed=160830;
LIBNAME home '/folders/myfolders/';

TITLE '1.a.';
PROC FREQ DATA=home.coalnuc;
	TABLES opinion * group / CHISQ;
	WEIGHT number;
RUN;

TITLE '1.b.';
PROC IML;
	c = {92, 35, 73};
	p = c/c[+];
	
	deviation = SQRT((p[1] * (1 - p[1]) + p[2] * (1 - p[2]) + 2 * p[1] * p[2]) / c[+]);
	critical_point = p[2] + QUANTILE('NORMAL', .95) * deviation;
	PRINT critical_point[LABEL='Critical point for p_coal'];
	
	result = p[1] < critical_point;
	PRINT result[LABEL='Accept Null Hypothesis?'];

TITLE '1.c. Confidence interval for percentage of environmentalists who support nuclear';
PROC SQL;
	CREATE TABLE coalnuc_1c AS
	SELECT group, SUM(number) AS number FROM (
		SELECT CASE WHEN group EQ 'E&C' AND OPINION EQ 'NUCLEAR' THEN 'E&C-NUCLEAR' ELSE 'NOT E&C-NUCLEAR' END AS group, *
		FROM home.coalnuc
	) GROUP BY group;

PROC FREQ DATA=coalnuc_1c;
   TABLES group / BINOMIAL(ac wilson EXACT) ALPHA=.05;
   WEIGHT number;
RUN;


TITLE '2.a';
PROC FREQ DATA=home.m_m ORDER=DATA;
      TABLES color / CHISQ testp=(.3, .2, .2, .1, .1, .1);
      WEIGHT number;
RUN;

TITLE '2.b and 2.c';
TITLE2 'Since P(χ² > obs) ≈ .0188 < .05 = α, reject H₀.';

PROC IML;
	samples = 100000;
	CALL RANDSEED(&seed);
	matrix = J(samples, 1);
	CALL RANDGEN(matrix, 'CHISQUARE', 5);
	pvalue = sum(matrix > 13.5405) / samples;
	PRINT pvalue;
RUN;


TITLE '3. Data does not appear to be normal';
PROC UNIVARIATE DATA=home.voltage;
	CLASS location;
	QQPLOT voltage / NORMAL(MU=est SIGMA=est COLOR=red L=2);

TITLE '3. Test for equality of means';
PROC NPAR1WAY WILCOXON CORRECT=NO DATA=home.voltage;
    VAR voltage;
	CLASS location;
RUN;

/* normality */
PROC UNIVARIATE DATA=home.vapor;
	QQPLOT exp / NORMAL(MU=est SIGMA=est COLOR=red L=2);
	QQPLOT calc / NORMAL(MU=est SIGMA=est COLOR=red L=2);

/* homoscedasticity */
PROC SQL;
	CREATE TABLE vapor_melt AS 
	SELECT calc AS value, 'CALC' AS label FROM home.vapor
    UNION ALL SELECT exp AS value, 'EXP' AS label FROM home.vapor;

PROC GLM DATA=vapor_melt;
	CLASS label;
   	MODEL value = label;
run;

TITLE '4. Vapor analysis';
PROC TTEST DATA=home.vapor;
	PAIRED exp*calc;
RUN;

TITLE '5.';
PROC IML;	
	CALL RANDSEED(&seed);
	samples = 10000;
	
	sample_sizes = {5, 10, 30, 50, 100};
	pop_proportions = {.01, .05, .25, .5, .9, .95};
/* 	demonstration of symmetry of p */
/* 	pop_proportions = {.05, .25, .5, .75, .95}; */

	/* p-values stored inside */
	results = J(NROW(sample_sizes), NROW(pop_proportions) + 1);
	margins = J(NROW(sample_sizes), NROW(pop_proportions) + 1);
	
	/* compute p-values for every combination of sample size and probability */
	DO i = 1 to NROW(sample_sizes);
		results[i, 1] = sample_sizes[i];
		margins[i, 1] = sample_sizes[i];
		
		DO j = 1 to NROW(pop_proportions);
			props = J(samples, 1);
			CALL RANDGEN(props, 'BINOMIAL', pop_proportions[j], sample_sizes[i]);
			
			props = props / sample_sizes[i];
			z_stat = QUANTILE('NORMAL', .975);

			interval = z_stat * SQRT(props # (1 - props) / sample_sizes[i]);
			results[i, j + 1] = SUM(ABS(props - pop_proportions[j]) < interval) / samples;
			margins[i, j + 1] = z_stat * SQRT(pop_proportions[j] * (1 - pop_proportions[j]) / sample_sizes[i]);
		END;
	END;
	
	/* pretty-print the table */
	colnames = 'n' // CHAR(pop_proportions);
	PRINT results[COLNAME=colnames LABEL='Coverage probability'];
	PRINT margins[COLNAME=colnames LABEL='Margin of error'];