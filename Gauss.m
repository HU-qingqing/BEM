function [ta,xl,wl] = Gauss(a,b,n)
ta = (b-a)/2;
tb = (b+a)/2;
%ta = (b(1)-a(1))/2;
%ta1 = (b(2)-a(2))/2;
%tb = (b(1)+a(1))/2;
%tb1 = (b(2)+a(2))/2;
%k = 2*pi;R=1;
if n == 1 
    xl = ta * 0 +tb;
    wl = 2;
elseif n == 2
    xl = ta*[0.57773503,-0.57773503]+tb;
    %xl1 = ta1*[0.57773503,-0.57773503]+tb1;
    wl = [1,1];
elseif n == 3
    xl = ta*[0.7745967,0,-0.7745967]+tb;
    %xl1 = ta1*[0.7745967,0,-0.7745967]+tb1;
    wl = [0.88888889,0.55555556,0.55555556];
elseif n == 4
    xl = ta*[0.3399810,-0.3399810,0.8611363,-0.8611363]+tb;
    %xl1 = ta1*[0.3399810,-0.3399810,0.8611363,-0.8611363]+tb1;
    wl = [0.6521452,0.6521452,0.3478548,0.3478548];
elseif n == 5
    xl = ta*[0.9061798,-0.9061798,0.5384693,-0.5384693,0]+tb;
    wl = [0.2369269,0.2369269,0.4786287,0.4786287,0.5688889];
elseif n == 6
    xl = ta*[0.9324695,-0.9324695,0.6612094,-0.6612094,0.2386192,-0.2386192]+tb;
    wl = [0.1713245,0.1713245,0.3607616,0.3607616,0.4679139,0.4679139];
elseif n == 7
    xl = ta*[0.9491079,-0.9491079,0.7415312,-0.7415312,0.4058452,-0.4058452,0]+tb;
    wl = [0.12948497,0.12948497,0.27970539,0.27970539,0.38183005,0.38183005,0.41795918];
end
    %xx = sqrt((x-xl).^2+xl1.^2);
    %xx = sqrt((x-cos(xl)).^2+sin(xl).^2);
    %xl = R*1i/4*besselh(0,1,k*xx);
    %val = sqrt(ta^2+ta1^2)*sum(xl.*wl);
    %val = ta*sum(xl.*wl);
end