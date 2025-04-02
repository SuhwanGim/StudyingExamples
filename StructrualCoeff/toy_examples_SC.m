%% ---------------------------
% 1. 데이터 생성
% ----------------------------
rng('default');    % 재현 가능한 결과를 위해 시드 고정
n = 100;           % 데이터 개수

% X1, X2 생성 (약간의 상관관계를 주기 위해 X2를 X1에 의존적으로 만듦)
X1 = randn(n,1);
X2 = 0.8*X1 + 0.5*randn(n,1);  % X1과 X2가 어느 정도 상관(공선성)을 갖게 함

% 종속변수 Y = 2*X1 + 3*X2 + 랜덤잡음
Y = 2*X1 + 3*X2 + 2*randn(n,1);

%% ---------------------------
% 2. 다중회귀분석 (비표준화 계수)
% ----------------------------
% 회귀분석을 위한 디자인행렬 X_full: 첫 열은 절편을 위한 ones
X_full = [ones(n,1), X1, X2];

% 비표준화 회귀계수 추정
[b, bint, ~, ~, stats] = regress(Y, X_full);
% b(1): 절편, b(2): X1에 대한 회귀계수, b(3): X2에 대한 회귀계수
% stats(1): R^2, stats(3): F값, stats(4): p값 등

% 예측값
Y_hat = X_full * b;

% ----------------------------
% 3. 표준화 베타계수 구하기
% ----------------------------
% Y, X1, X2를 각각 표준화한 다음 동일하게 회귀분석
b_std = regress(zscore(Y), [ones(n,1), zscore(X1), zscore(X2)]);
% b_std(2), b_std(3)이 표준화된 베타계수에 해당

% ----------------------------
% 4. Structure coefficient 계산
% ----------------------------
% Structure coefficient = corr(X_i, Y_hat)
structCoeff_X1 = corr(X1, Y_hat);
structCoeff_X2 = corr(X2, Y_hat);

% ----------------------------
% 5. 결과 확인
% ----------------------------
fprintf('----- 비표준화 회귀계수 (b) -----\n');
disp(b);

fprintf('----- 표준화 베타계수 (b_std) -----\n');
disp(b_std);

fprintf('----- Structure Coefficients -----\n');
fprintf('corr(X1, Y_hat) = %.3f\n', structCoeff_X1);
fprintf('corr(X2, Y_hat) = %.3f\n', structCoeff_X2);

fprintf('----- R^2 (설명력) -----\n');
fprintf('R^2 = %.3f\n', stats(1));
