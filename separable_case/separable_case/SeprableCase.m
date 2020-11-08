%Nora Basha- 12/03/2019- 
%ECE 599-Large Scale convex/non-convex optimization
% SVM- HW#3- Q4-Seprable case
%====================================================================================
%Loading training and testing data
S = load('C:\Fall 2019\Optimization\HW\separable_case\separable_case\train_separable.mat');
A=S.A;
B=S.B;
S2 = load('C:\Fall 2019\Optimization\HW\separable_case\separable_case\test_separable.mat');
test=S2.X_test;
True_labels=S2.true_labels;
%Finding the closest two points u,v of the two convex hull of class A and B
cvx_begin
    variable u(2)
    variable v(2)
    minimize(0.5*(A'*u-B'*v)' *(A'*u-B'*v))
    subject to
       transpose(ones ([2,1]))*u==1
       transpose(ones ([2,1]))*v==1
       u>=0
       v>=0
cvx_end
%Finding the hyperplane passing by the middle point of the closest two
%points u, v
cvx_begin
    variable w(2)
    variable a(1)
    variable b(1)
    minimize(0.5*w'*w-(a-b))
    subject to
       A'*w-a>=0
      -B'*w+b>=0
cvx_end
%Visualizing 
y=-100:0.2:100;
y1=zeros(1, length(y));
y2=zeros(1, length(y));
Classifier=zeros(1, length(y));
for i=1:length(y)
y1(i)=y(i);
y2(i)=y(i);
Classifier(i)= (w(1)*y1(i))+(w(2)*y2(i))-(a+b)/2;
end
%The training data and the classifier
figure
hold on
plot3(y1,y2,Classifier,'DisplayName','Classifier')
scatter(A(1,:),A(2,:),'DisplayName','Class A')
scatter(B(1,:),B(2,:), 'DisplayName','Class B')
hold off
%The training data and the classifier form a different angle
figure
hold on
plot3(y1,Classifier,y2,'DisplayName','Classifier Plan view with training data')
scatter(A(1,:),A(2,:),'DisplayName','Class A')
scatter(B(1,:),B(2,:), 'DisplayName','Class B')
hold off
%The testing data and the classifier
figure
hold on
plot3(y1,y2,Classifier, 'DisplayName','Classifier')
scatter(test(1,:),test(2,:),'DisplayName','testing data')
hold off
%The testing data and the classifier from a different angle
figure
hold on
plot3(y1,Classifier,y2, 'DisplayName','Classifier Plan view with training data')
scatter(test(1,:),test(2,:), 'DisplayName','Testing data')
hold off
%Calculating error and accuracy
predicted_labels=sign(w'*test-(a+b)/2);
error1=predicted_labels-True_labels;% same signs will cancel remaining the misclassified records
index= find(error1~=0);
error=sum(error1(index))/length(error1)
accuracy_percentage=(1-error)*100



