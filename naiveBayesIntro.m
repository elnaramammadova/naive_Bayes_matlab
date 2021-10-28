%% 
% likes shortbread, likes lager, eats porridge, watches England playing
% football, scottish/english

% The right-most element in the vector denotes the class that we want to predict, and it
% can take two values: 1 for Scottish, 0 for English

X = [ 0 0 1 1 0 ;
1 0 1 0 0 ;
1 1 0 1 0 ;
1 1 0 0 0 ;
0 1 0 1 0 ;
0 0 1 0 0 ;
1 0 1 1 1 ;
1 1 0 1 1 ;
1 1 1 0 1 ;
1 1 1 0 1 ;
1 1 1 1 1 ;
1 0 1 0 1 ;
1 0 0 0 1 ];
%% 

Y = X(:,5);  % scottish/english label
X = X(:,1:4)'; % binary attributes (likes shortbread, likes lager, 
% eats porridge, watches England playing football)
% apply  _'_ computes the complex conjugate transpose to be able to do
% inner product computations

%% 

% class priors (number of samples in the class) / (size/total number of the samples)
% for two priors, S(scotland) and E(english)
% probability of being scottish given it is in Y p(S|Y) 
% S(# of scottish)=1 (where values in Y vector equal to 1 - sum(Y)
% probability of being english give it is in Y p(E|Y) 
% E(# of english)=0 (where values in Y vector equal to 0 - sum(1-Y)

% we sum the products of all the elements of X times Y (Y=1) and 1-Y (Y=0)
% 

pS = sum (Y)/size(Y,1);     % all rows with Y = 1 
pE = sum(1 - Y)/size(Y,1);  % all rows with Y = 0

%% 
% now on to max log likelihood

% to find the likelihood of other classifiers given that the person is
% either scottish or english you do sum product of X and Y or X and Y-1

% e.g. the likelihood "likes shortbread" binary attribute by class label "scottish"
% X(1,:)*Y(:,1) = 7    (instances where X=1, Y=1)
% X(1,:)  0     1     1     1     0     0     1     1     1     1     1     1     1
% Y(:,1)' 0     0     0     0     0     0     1     1     1     1     1     1     1

% similary the likelihood "likes shortbread" binary attribute by class
% label "english"
% X(1,:)*(1-Y) = 3  (instances where X=1, Y=1) 
% X(1,:)  0     1     1     1     0     0     1     1     1     1     1     1     1
% (1-Y)'  1     1     1     1     1     1     0     0     0     0     0     0     0


% X is a 4x13 vector, Y is a 13x1 vector
% X*Y will be a 4x1 vector, since the matrix product will return inner 
% product C(i,j) = A(i,:)*B(:,j)


phiS = X * Y / sum(Y);  % all instances for which attrib phi(i) and Y are both 1
                        % P (X/Y=1) where both X binary attributes and Y
                        % labels return 1 (scottish)
                        % *(phi(i) here is Y - the label scottish/english
                        
% X * Y
%     7
%     4
%     5
%     3

% sum(Y) = 7

% X * Y / sum(Y)
%    1.0000
%    0.5714
%    0.7143
%    0.4286                        
                        
% So "X * Y / sum(Y)" is the mean (mu) of the attribute value w.r.t. class label
% (scottish) which is also the probability of X=1 given Y=1.
              
phiE = X * (1-Y) / sum(1-Y) ;  % all instances for which attrib phi(i) = 1 and Y = 0
                               % P(X/Y=0) where X binary attributes returns 1 and Y 
                               % labels return 0 (english)
                               
% X * (1-Y) / sum(1-Y)
%    0.5000
%    0.5000
%    0.5000
%    0.5000

% So "X * Y / sum(Y)" is the mean (mu) of the attribute value w.r.t. class
% label (english) which is also the probability of X=1 given Y=0.
                       
x=[1 0 1 0]';  % test point ---> shortbread yes, lager no, porridge yes, watches england no
%% 
              
% Bernoulli distribution: f = (p^k)((1-p)^(1-k))
% Remember our x here is a vector of binary values, so that 
% phiS.^x.*(1-phiS).^(1-x) below will yeild a vector with each element being 
% a probability. 
% Then prod can be used to multiply all the values in x, as follows: 

% Probability of observing test point vector, given class label scottish
pxS = prod(phiS.^x.*(1-phiS).^(1-x));
%    phiS       x   phiS.^x     1-phiS  1-x (1-phiS).^(1-x) phiS.^x.*(1-phiS).^(1-x)     
%    1.0000     1   1.0000      0       0   1.0000          1.0000
%    0.5714     0   1.0000      0.4286  1   0.4286          0.4286
%    0.7143     1   0.7143      0.2857  0   1.0000          0.7143
%    0.4286     0   1.0000      0.5714  1   0.5714          0.5714
% prod(phiS.^x.*(1-phiS).^(1-x)) = 1 * 0.4286 * 0.7143 * 0.5714 = 0.1749 
% pxS = 0.1749 

% Probability of observing test point vector-x, given class label english
pxE = prod(phiE.^x.*(1-phiE).^(1-x));
%    phiE      x         phiE.^x   1-phiE    1-x       (1-phiE).^(1-x)  phiE.^x.*(1-phiE).^(1-x)
%    0.5000    1.0000    0.5000    0.5000         0    1.0000           0.5000
%    0.5000         0    1.0000    0.5000    1.0000    0.5000           0.5000
%    0.5000    1.0000    0.5000    0.5000         0    1.0000           0.5000
%    0.5000         0    1.0000    0.5000    1.0000    0.5000           0.5000
% prod(phiE.^x.*(1-phiE).^(1-x)) = 0.5000 * 0.5000 * 0.5000 * 0.5000 = 0.0625
% pxE = 0.0625 
%% 

% Bayesian formula

% Probability of observing class label scottish, given test point vector-x
% p(x|S) = p(x|S)*p(S)/p(x|S)*p(S) + p(x|E)*p(E)
pxSF = (pxS * pS ) / (pxS * pS + pxE * pE) %P(Y=1|X)=0.76555 

% Probability of observing class label english, given test point vector-x
% p(x|E) = p(x|E)*p(E)/p(x|E)*p(E) + p(x|S)*p(S)

pxEF = (pxE * pE ) / (pxS * pS + pxE * pE) %P(Y=0|X)=0.23445

%% 
% Change the test point on line 25 to use different values and check the results.

T =[0 0 0 0;
    0 0 0 1;
    0 0 1 0;
    0 0 1 1;
    0 1 0 0;
    0 1 0 1;
    0 1 1 0;
    0 1 1 1;
    1 0 0 0;
    1 0 0 1;
    1 0 1 0;
    1 0 1 1;
    1 1 0 0;
    1 1 0 1;
    1 1 1 0;
    1 1 1 1];

for i = 1:16 
     x = T(i,:)'; % transpose, so x*(1-x) is 4x1 vector
     pxS = prod(phiS.^x.*(1-phiS).^(1-x));
     pxE = prod(phiE.^x.*(1-phiE).^(1-x));
     pxSF = (pxS * pS ) / (pxS * pS + pxE * pE);
     pxEF = (pxE * pE ) / (pxS * pS + pxE * pE);
     disp("% Test point:" + i)
     disp(T(i,:))
     disp("% pxSF = " + string(pxSF) + ", psEF = " + string(pxEF))
end;

% Test point:1
%     0     0     0     0   -- nothing - english

% pxSF = 0, psEF = 1
% Test point:2
%     0     0     0     1   -- england - english

% pxSF = 0, psEF = 1
% Test point:3
%     0     0     1     0   -- porridge - english

% pxSF = 0, psEF = 1
% Test point:4
%     0     0     1     1   -- porridge, england - english

% pxSF = 0, psEF = 1
% Test point:5
%     0     1     0     0   -- lager - english

% pxSF = 0, psEF = 1
% Test point:6
%     0     1     0     1   -- lager, england - english

% pxSF = 0, psEF = 1
% Test point:7
%     0     1     1     0   -- lager, porridge - english

% pxSF = 0, psEF = 1
% Test point:8
%     0     1     1     1   -- lager, porridge, england - english 

% pxSF = 0, psEF = 1
% Test point:9
%     1     0     0     0   -- shortbread - scottish

% pxSF = 0.56637, psEF = 0.43363
% Test point:10
%     1     0     0     1   -- shortbread, england - english

% pxSF = 0.49485, psEF = 0.50515
% Test point:11
%     1     0     1     0   -- shortbread, porridge - scottish

% pxSF = 0.76555, psEF = 0.23445
% Test point:12
%     1     0     1     1   -- shortbread, porridge, england - scottish

% pxSF = 0.71006, psEF = 0.28994
% Test point:13
%     1     1     0     0   -- shortbread, lager - scottish

% pxSF = 0.63524, psEF = 0.36476
% Test point:14
%     1     1     0     1   -- shortbread, lager, england - scottish

% pxSF = 0.56637, psEF = 0.43363
% Test point:15
%     1     1     1     0   -- shortbread, lager, porridge - scottish

% pxSF = 0.81321, psEF = 0.18679
% Test point:16
%    1     1     1     1   -- shortbread, lager, porridge, england - scottish

% pxSF = 0.76555, psEF = 0.23445