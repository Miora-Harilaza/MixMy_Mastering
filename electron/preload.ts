import { contextBridge, ipcRenderer } from 'electron';
contextBridge.exposeInMainWorld('mixmaster',{
  openAudio:()=>ipcRenderer.invoke('audio:open'), saveAudio:(name:string)=>ipcRenderer.invoke('audio:save',name), exportAudio:(source:string)=>ipcRenderer.invoke('audio:export',source),
  analyze:(input:string)=>ipcRenderer.invoke('audio:analyze',input), preparePlayback:(input:string)=>ipcRenderer.invoke('audio:preview',input), master:(args:{input:string;output:string;preset:string})=>ipcRenderer.invoke('audio:master',args), cancel:()=>ipcRenderer.invoke('audio:cancel'),
  onProgress:(callback:(progress:{stage:string;percent:number})=>void)=>{const handler=(_event:unknown,data:{stage:string;percent:number})=>callback(data);ipcRenderer.on('audio:progress',handler);return()=>ipcRenderer.removeListener('audio:progress',handler);}
});
