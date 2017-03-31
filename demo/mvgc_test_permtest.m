%% MVGC permutation test demo
%
% Demonstrates permutation significance testing with MVGC on generated VAR data
% for a 5-node network with known causal structure (see <var5_test.html
% |var5_test|>). Pairwise-conditional Granger causalities are estimated and
% significance tested using both the theoretical and permutation test null
% distributions. The theoretical and permutation test null distributions for all
% pairs are plotted together and compared using a Kolmogorov-Smirnov test (see
% <matlab:doc('kstest') |kstest|>).
%
%% References
%
% [1] L. Barnett and A. K. Seth,
% <http://www.sciencedirect.com/science/article/pii/S0165027013003701 The MVGC
%     Multivariate Granger Causality Toolbox: A New Approach to Granger-causal
% Inference>, _J. Neurosci. Methods_ 223, 2014
% [ <matlab:open('mvgc_preprint.pdf') preprint> ].
%
% [2] M. J. Anderson and J. Robinson, Permutation tests for linear models,
% _Aust. N. Z. J. Stat._, 43(1), 2001.
%
% (C) Lionel Barnett and Anil K. Seth, 2012. See file license.txt in
% installation directory for licensing terms.
%
%% Parameters

ntrials   = 144;     % number of trials
nobs      = 2033;   % number of observations per trial
nvars     = 7;
nperms    = 100;    % number of permutations for permutation test
bsize     = [];     % permutation test block size: empty for automatic (uses model order)

morder    = 'AIC';  % model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
regmode   = 'OLS';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
icregmode = 'LWR'; 
acmaxlags = 1000;   % maximum autocovariance lags (empty for automatic calculation)
momax     = 10;  

tstat     = '';     % statistical test for MVGC: 'F' for Granger's F-test, 'chi2' for Geweke's chi2 test or leave empty for default
alpha     = 0.001;   % significance level for all statistical tests
mhtc      = 'FDR';  % multiple hypothesis test correction (see routine 'significance')

fs        = 1017;    % sample rate (Hz)
fres      = [];     % frequency resolution (empty for automatic calculation)
frange    = [0,100];

seed      = 0;      % random seed (0 for unseeded)

%% Generate VAR data

rawdata = h5read('/groups/Projects/P1299/Analysis/temp_directory/item0_pair_7nodes.hf', '/data');
ndim = numel(size(rawdata));
file_array = permute(rawdata,[ndim:-1:1]);
X = file_array;

%% VAR model estimation and autocovariance calculation

% Calculate information criteria up to specified maximum model order.

ptic('\n*** tsdata_to_infocrit\n');
[AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
ptoc('*** tsdata_to_infocrit took ');

% Plot information criteria.

figure(1); clf;
plot_tsdata([AIC BIC]',{'AIC','BIC'},1/fs);
title('Model order estimation');

% amo = size(AT,3); % actual model order

fprintf('\nbest model order (AIC) = %d\n',moAIC);
fprintf('best model order (BIC) = %d\n',moBIC);
% fprintf('actual model order     = %d\n',amo);

% Select model order.

if     strcmpi(morder,'actual')
    morder = amo;
    fprintf('\nusing actual model order = %d\n',morder);
elseif strcmpi(morder,'AIC')
    morder = moAIC;
    fprintf('\nusing AIC best model order = %d\n',morder);
elseif strcmpi(morder,'BIC')
    morder = moBIC;
    fprintf('\nusing BIC best model order = %d\n',morder);
else
    fprintf('\nusing specified model order = %d\n',morder);
end


% Calculate VAR model

ptic('*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(X,morder,regmode);
ptoc;

% Check for failed regression

assert(~isbad(A),'VAR estimation failed');

% Now calculate autocovariance according to the VAR model, to as many lags
% as it takes to decay to below the numerical tolerance level, or to acmaxlags
% lags if specified (i.e. non-empty).

ptic('*** var_to_autocov... ');
[G,res] = var_to_autocov(A,SIG,acmaxlags);
ptoc;

% Report and check for errors.

fprintf('\nVAR check:\n'); disp(res); % report results...
assert(~res.error,'bad VAR');         % ...and bail out if there are errors

%% Granger causality estimation

% Calculate time-domain pairwise conditional causalities.

ptic('*** autocov_to_pwcgc... ');
F = autocov_to_pwcgc(G);
ptoc;

% Theoretical significance test (adjusting for multiple hypotheses).

pval_t = mvgc_pval(F,morder,nobs,ntrials,1,1,nvars-2,tstat);
sig_t  = significance(pval_t,alpha,mhtc);

%% Permutation test

ptic('\n*** tsdata_to_mvgc_pwc_permtest\n');
FNULL = permtest_tsdata_to_pwcgc(X,morder,bsize,nperms,regmode,acmaxlags);
ptoc('*** tsdata_to_mvgc_pwc_permtest took ',[],1);

% (We should really check for permutation estimates here.)

% Permutation test significance test (adjusting for multiple hypotheses).

pval_p = empirical_pval(F,FNULL);
sig_p  = significance(pval_p,alpha,mhtc);

%% Plot causalities, p-values, significance and cdfs

figure(1); clf;
subplot(2,3,1);
plot_pw(F);
title('Pairwise-conditional GC');
subplot(2,3,2);
plot_pw(pval_t);
title({'p-values';'(theoretical)'});
subplot(2,3,3);
plot_pw(sig_t);
title({['Significant at p = ' num2str(alpha)];'(theoretical)'});
subplot(2,3,5);
plot_pw(pval_p);
title({'p-values';'(perm test)'});
subplot(2,3,6);
plot_pw(sig_p);
title({['Significant at p = ' num2str(alpha)];'(perm test)'});

% Plot empirical vs theoretical null cdfs, perform Kolmogorov-Smirnov tests on
% null/theoretical distributions.

figure(2); clf;
pval_ks = nan(nvars); % KS test p-value
fmax = max(FNULL(:));
k = 0;
for i = 1:nvars
    for j = 1:nvars
        k = k+1;
        if i ~= j
            FN = sort(FNULL(:,i,j));
            PN = empirical_cdf(FN,FNULL(:,i,j));
            PT = 1-mvgc_pval(FN,morder,nobs,ntrials,1,1,nvars-2,tstat); % theoretical cdf = 1 - p-value
            subplot(nvars,nvars,k);
            plot(FN,[PN PT]);
            title(sprintf('%d -> %d',j,i));
            xlim([0 fmax]);
            [~, pval_ks(i,j)] = kstest(FN,[FN PT]); % 2-sided KS test of empirical vs. theoretical null distribution
        end
    end
end

% KS significance test (adjusting for multiple hypotheses).

sig_ks = significance(pval_ks,alpha,mhtc); % 1 = distributions are significantly different at level alpha

fprintf('\nKS-test p-values:\n\n'); disp(pval_ks);
fprintf('KS significance test; ones indicates that permutation and theoretical\nnull distributions are significantly different at p-value %g; \n\n',alpha); disp(sig_ks);

%%
% <mvgc_demo_permtest.html back to top>
