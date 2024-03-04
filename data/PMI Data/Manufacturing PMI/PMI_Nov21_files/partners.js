(function(){var i,q=[];
function wait(p){
  if (p||document.body){q.forEach(add);clearInterval(i)}
  else if (!i){i = setInterval(wait, 20)}
}
function add(s,p){
  if(p||document.body){(p||document.body).appendChild(s)}
  else{q.push(s);wait()}
}
function addScript(s,o){var t=document.createElement('script');t.async=1;t.src=s;add(Object.assign(t,o||{}))}
function addImg(s,d){var t=document.createElement('img');t.src=s;t.style.cssText='height:0;width:0;border:0;display:none;';t.alt=d;add(t)}
function addIframe(s,d,o,p){var t=document.createElement('iframe');t.src=s;t.style.cssText='height:0;width:0;display:none;visibility:hidden;';t.title=d;add(Object.assign(t,o||{}),p)}

window._oiqq = window._oiqq || [];
_oiqq.push(['oiq_addPageCat','Computer']);
_oiqq.push(['oiq_addPageLifecycle', 'inte']);
_oiqq.push(['oiq_doTag']);

addScript('https://px.owneriq.net/stas/s/sholic.js');


(function (w,d) {
  _ml = w._ml || {};
  _ml.nq = w._ml.nq || [];
  _ml.nq.push(['track', '51840']);
  var s, cd, tag; cd = new Date();
  tag = addScript('https://ml314.com/taglw.aspx?' + cd.getDate() + cd.getMonth());
})(window,document);

window._comscore = window._comscore || [];
_comscore.push({ c1: "7", c2: "19376307" ,c3: "1" });
addScript("https://sb.scorecardresearch.com/beacon.js");

(function(){try{var s,w=window.top;w.Tynt=w.Tynt||[];
w.Tynt.push('sh!sh');addScript('https://cdn.tynt.com/afsh.js');
}catch(e){}})();

})();