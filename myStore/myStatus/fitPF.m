clc;clear;close all;

load('D:\Ye Li''s Code\强化学习测试\重写\集成\myStatus\optimization_moo.mat')
x=results.paretoFront(:,1);
y=results.paretoFront(:,2);

num = 8;

plot(x,y,'o');
p=polyfit(x,y,num);
x1=min(x):0.001:max(x);
y1=polyval(p,x1);
hold on
plot(x1,y1)