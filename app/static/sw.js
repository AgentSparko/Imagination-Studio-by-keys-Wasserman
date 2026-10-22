"use strict";
// The studio's service worker: it shows the notifications the studio sends (app/push.py), and a tap opens the studio.
// It does not cache or answer any request: the page loads from the studio as before.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
self.addEventListener("push", event => {
  let said = {};
  try { said = event.data ? event.data.json() : {}; } catch { said = { body: event.data ? event.data.text() : "" }; }
  event.waitUntil(self.registration.showNotification(said.title || "Imagination Studio", {
    body: said.body || "", icon: "brand/app-icon-192.png", badge: "brand/app-icon-192.png", tag: said.tag || "studio", data: { url: said.url || "/" } }));
});
self.addEventListener("notificationclick", event => {
  event.notification.close();
  const url = new URL((event.notification.data && event.notification.data.url) || "/", self.location.origin).href;
  event.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then(open => {
    const here = open.find(client => client.url.startsWith(self.location.origin));
    if (here) return (here.navigate ? here.navigate(url).catch(() => here) : Promise.resolve(here)).then(client => (client || here).focus());
    return self.clients.openWindow(url);
  }));
});
