/* eslint-disable no-unused-vars */
// @ts-nocheck
"use_strict"

class WindowMode
{
    static FitScreen = "fit_screen";
    static RealPixels = "real_pixels";
}

/** @type {HTMLCanvasElement} */
let canvas;
let ZOOMX = 1;
let ZOOMY = 1;
let WORLDW = 600;
let WORLDH = 600;
let backBuffer;
let upscaledBackBuffer;
let smoothing = false;
let playerInteracted = false;
let allSounds = [];
let pendingStuffToLoad = 0;
let audioHackPlayedSilentSoundToPreventLag = false;
let initializedPostLoad = false;
let mouseScreenX = 0;
let mouseScreenY = 0;
let mousex = 0;
let mousey = 0;
let mousePressed = false;
let mouseJustPressed = false;
let mousePressedRight = false;
let mouseJustPressedRight = false;
let screenMode = WindowMode.FitScreen;
let keysJustPressed = [];
let keysPressed = [];
let shiftIsPressed = false;
let lastUpdateTime = Date.now();
let timeElapsed = 0;

class UpdatePhase
{
    static Init= "init";
    static Loading = "loading";
    static DoneLoading = "done_loading";
    static Updating = "updating";
}

class Strip
{
    constructor()
    {
        /** @type {StripFrame[]} */
        this.frames = [];
    }
}

class StripFrame
{
    constructor()
    {
        this.img = null;
        this.rect = new Rect();
        this.pivotx = 0;
        this.pivoty = 0;
    }
}

function loadSoundStreamed(path)
{
    let ret = new Audio(path);
    ret.preload = "none";
    // ret.load();
    ret.onloadeddata = function()
    {
        allSounds.push(ret);
    };
    return ret;    
}

function loadSound(path)
{
    pendingStuffToLoad += 1;
    let ret = new Audio(path);
    ret.preload = "auto";
    ret.load();
    ret.onloadeddata = function()
    {
        // if(!audioHackPlayedSilentSoundToPreventLag)
        // {
        //     audioHackPlayedSilentSoundToPreventLag = true;
        //     let old = ret.volume;
        //     ret.volume = 0.00001;
        //     ret.play();
        //     ret.volume = old;
        // }
        allSounds.push(ret);
        pendingStuffToLoad -= 1;
    };
    return ret;
}

function loadImage(path, fnAfterLoad)
{
    pendingStuffToLoad += 1;
    let ret = new Image();
    ret.src = "data/"+path;
    ret.onload = function()
    {
        pendingStuffToLoad -= 1;
        if(fnAfterLoad != null) fnAfterLoad(ret);
    };
    return ret;    
}

function loadStrip(path, cellw, cellh, pivotx, pivoty, colorMultiplier = 0xffffff, fnAfterLoad)
{
    let ret = new Strip();
    loadImage(path, function(img)
    {
        let framesW = Math.floor(img.width / cellw);
        let framesH = Math.floor(img.height / cellh);
        for(let i = 0; i < framesH; i++) 
        {
            for(let j = 0; j < framesW; j++) 
            {
                let frame = new StripFrame();
                frame.pivotx = pivotx;
                frame.pivoty = pivoty;
                frame.rect.set(j * cellw, i * cellh, cellw, cellh);
                ret.frames.push(frame);

                let frameImage = document.createElement("canvas");
                frameImage.width = cellw;
                frameImage.height = cellh;
                let ctx = frameImage.getContext("2d");
                ctx.drawImage(img, frame.rect.x, frame.rect.y, frame.rect.w, frame.rect.h, 0, 0, frame.rect.w, frame.rect.h);
                if(colorMultiplier != 0xffffff)
                {
                    let r = (colorMultiplier >> 16 & 0xff)/0xff;
                    let g = (colorMultiplier >> 8 & 0xff)/0xff;
                    let b = (colorMultiplier >> 0 & 0xff)/0xff;
                    let imgData = ctx.getImageData(0, 0, cellw, cellh);
                    for(let i = 0; i < imgData.data.length; i += 4) 
                    {
                        imgData.data[i + 0] *= r;
                        imgData.data[i + 1] *= g;
                        imgData.data[i + 2] *= b;
                    }
                    ctx.putImageData(imgData, 0, 0);                
                }
                frame.img = frameImage;
                // console.log("r "+(j + i*framesW)+" es "+frame.rect.toString());
            }
        }
        if(fnAfterLoad != null) fnAfterLoad(ret);
    });
    return ret;
}

function createFontVariant(otherFont, colorMultiplier)
{
    let ret = new BitmapFont();
    ret.chars = otherFont.chars;
    ret.glyphs = otherFont.glyphs;
    ret.lineh = otherFont.lineh;
    ret.max_char_height = otherFont.max_char_height;
    ret.char_sep = otherFont.char_sep;
    ret.hasCapitalization = otherFont.hasCapitalization;
    ret.strip = new Strip();
    for(let otherFrame of otherFont.strip.frames)
    {
        let f = new StripFrame();
        f.rect = otherFrame.rect;
        f.pivotx = otherFrame.pivotx;
        f.pivoty = otherFrame.pivoty;
        f.img = createElement("canvas");
        f.img.width = otherFrame.img.width;
        f.img.height = otherFrame.img.height;
        let ctx = f.img.getContext("2d");
        ctx.drawImage(f.img, 0, 0);
        ret.strip.frames.push(f);
    }
    return ret;
}

function createFont(image, cellw, cellh, pivotx, pivoty, charMap, hasCapitalization, colorMultiplier = 0xffffff, fnAfterLoad = null)
{
    let strip = new Strip();
    strip.lo

    let ret = new BitmapFont();
    ret.loadFromStrip()

    loadStrip(imagePath, cellw, cellh, pivotx, pivoty, colorMultiplier, function(strip)
    {
        ret.loadFromStrip(strip, charMap, hasCapitalization);
        if(fnAfterLoad != null) fnAfterLoad(ret);
    });
    return ret;
}

function loadFont(imagePath, cellw, cellh, pivotx, pivoty, charMap, hasCapitalization, colorMultiplier = 0xffffff, fnAfterLoad = null)
{
    let ret = new BitmapFont();
    loadStrip(imagePath, cellw, cellh, pivotx, pivoty, colorMultiplier, function(strip)
    {
        ret.loadFromStrip(strip, charMap, hasCapitalization);
        if(fnAfterLoad != null) fnAfterLoad(ret);
    });
    return ret;
}

function fitCanvas()
{
    screenMode = isMobile() ? WindowMode.FitScreen : WindowMode.FitScreen;
    let windowW = window.innerWidth;
    let windowH = window.innerHeight;
    let targetW = windowW;
    let targetH = windowH;
    let widthOverHeight = backBuffer.width / backBuffer.height;
    if(screenMode == WindowMode.FitScreen)
    {
        if(windowH > windowW)
        {
            targetW = windowW;
            targetH = targetW / widthOverHeight;    
        }
        else
        {
            targetH = windowH * 0.975;
            targetW = targetH * widthOverHeight;    
        }
        smoothing = true;
    }
    else
    if(screenMode == WindowMode.RealPixels)
    {
        smoothing = false;
        targetW = backBuffer.width * ZOOMX;// / window.devicePixelRatio;
        targetH = backBuffer.height * ZOOMY;// / window.devicePixelRatio;
    }
    canvas.width = targetW;
    canvas.height = targetH;
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.bottom = "0";
    canvas.style.left = "0";
    canvas.style.right = "0";
    canvas.style.margin = "auto";
    canvas.style.imageRendering = "pixelated;crisp-edges";
}

function onLoadPage()
{
    document.addEventListener("focus",
        (event) => 
        {
            shiftIsPressed = false;
        },
        true,
      );

      document.addEventListener(
        "blur",
        (event) => {
            shiftIsPressed = false;
        },
        true,
      );

    document.addEventListener("touchstart", 
        function (evt)
        {
            evt.preventDefault();
            mousePressed = true;
            mouseJustPressed = true;
            var rect = canvas.getBoundingClientRect();
            for(var i = 0; i < evt.touches.length; i++)
            {
                let touch = evt.touches[i];
                mouseScreenX = touch.clientX - rect.x;
                mouseScreenY = touch.clientY - rect.y;
            }
        },
    { passive: false });

    document.addEventListener("touchmove",
        function (evt)
        {
            evt.preventDefault();
            var rect = canvas.getBoundingClientRect();
            for(var i = 0; i < evt.touches.length; i++)
            {
                let touch = evt.touches[i];
                mouseScreenX = touch.clientX - rect.x;
                mouseScreenY = touch.clientY - rect.y;
            }
        },
        { passive: false });

    document.addEventListener("touchend",
        function (evt)
        {
            evt.preventDefault();
            activatePlayerInteraction();
            mousePressed = false;
        },
        { passive: false });


    document.onmousemove = function (me) 
    {
        var rect = canvas.getBoundingClientRect();
        mouseScreenX = me.clientX - rect.x; 
        mouseScreenY = me.clientY - rect.y;
    };

    document.onclick = function(me) {activatePlayerInteraction();}

    document.addEventListener('contextmenu', (event) => 
    {
        // Prevent the default context menu from appearing
        event.preventDefault();
    });

    document.onmousedown = function (e) 
    {
        shiftIsPressed = e.shiftKey;
        if(e.button == 0)
        {
            mousePressed = true;
            mouseJustPressed = true;
        }
        else
        if (e.which && e.which === 3 || e.button === 2)
        {
            mousePressedRight = true;
            mouseJustPressedRight = true;
        }
        activatePlayerInteraction();
    };

    document.onmouseup = function (e) 
    {
        // console.log("mup button:"+me.button+" buttons:"+me.buttons + " target: "+me.target);
        if(e.button == 0)
        {
            mousePressed = false;
        }
        else 
        if (e.which && e.which === 3 || e.button === 2)
        {
            mousePressedRight = false;
        }
    };

    document.onkeydown = function (keyEvent) 
    { 
        keysJustPressed.push(keyEvent.key);
        keysPressed.push(keyEvent.key);
        shiftIsPressed = keyEvent.shiftKey;
    };

    document.onkeyup = function (keyEvent)
    {
        keysPressed = keysPressed.filter(k => k != keyEvent.key);
        shiftIsPressed = keyEvent.shiftKey;
    };

    // schedule stuff to load
    onUpdate(UpdatePhase.Init, 0);

    document.body.style.margin = "0px 0px 0px 0px";
    document.body.style.backgroundColor = "black";

    backBuffer = document.createElement("canvas");
    backBuffer.width = WORLDW;
    backBuffer.height = WORLDH;

    upscaledBackBuffer = document.createElement("canvas");
    upscaledBackBuffer.width = backBuffer.width * 2;
    upscaledBackBuffer.height = backBuffer.height * 2;

    canvas = document.createElement("canvas");
    fitCanvas();
    document.body.appendChild(canvas);
    window.onresize = fitCanvas;

    window.requestAnimationFrame(onInternalUpdate);
}

function onInternalUpdate(now)
{
    let dt = (Date.now() - lastUpdateTime) / 1000;
    dt = Math.min(dt, 1/60);
    lastUpdateTime = Date.now();
    timeElapsed += dt;

    mousex = (mouseScreenX / canvas.width) * backBuffer.width;
    mousey = (mouseScreenY / canvas.height) * backBuffer.height;

    if(pendingStuffToLoad > 0)
    {
        onUpdate(UpdatePhase.Loading, dt);
    }
    else
    {
        if(!initializedPostLoad)
        {
            onUpdate(UpdatePhase.DoneLoading, dt);
            initializedPostLoad = true;
        }
        onUpdate(UpdatePhase.Updating, dt);
    }

    // flip backbuffer
    let ctx = get2DContext(canvas);
    ctx.imageSmoothingEnabled = smoothing;
    // upscale the backbuffer if stretching
    if(screenMode == WindowMode.FitScreen)
    {
        let doubleCtx = upscaledBackBuffer.getContext("2d");
        doubleCtx.imageSmoothingEnabled = false;
        doubleCtx.drawImage(backBuffer, 0, 0, backBuffer.width, backBuffer.height, 0, 0, upscaledBackBuffer.width, upscaledBackBuffer.height);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(upscaledBackBuffer, 0, 0, upscaledBackBuffer.width, upscaledBackBuffer.height, 0, 0, canvas.width, canvas.height);
    }
    else
    {
        ctx.drawImage(backBuffer, 0, 0, backBuffer.width, backBuffer.height, 0, 0, canvas.width, canvas.height);
    }

    keysJustPressed = [];
    mouseJustPressed = false;
    mouseJustPressedRight = false;
    window.requestAnimationFrame(onInternalUpdate);
}

function get2DContext(img)
{
    return img.getContext("2d");
}

class Rect
{
    constructor(x = 0, y = 0, w = 0, h = 0)
    {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }

    floor()
    {
        this.x = Math.floor(this.x);
        this.y = Math.floor(this.y);
        this.w = Math.floor(this.w);
        this.h = Math.floor(this.h);
    }

    contains(x, y)
    {
        return x >= this.x && x < this.right() && y >= this.y && y < this.bottom();
    }

    setR(r)
    {
        this.w = r - this.x;
    }

    setB(b)
    {
        this.h = b - this.y;
    }

    set(x, y, w, h)
    {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }

    copyFrom(other)
    {
        this.x = other.x;
        this.y = other.y;
        this.w = other.w;
        this.h = other.h;
    }

    resizeBy(width, height)
    {
        this.w -= width;
        this.h -= height;
        this.x += width/2;
        this.y += height/2;
    }

    enclose(other)
    {
        this.x = Math.min(this.x, other.x);
        this.y = Math.min(this.y, other.y);
        this.w = Math.max(this.right(), other.right()) - this.x;
        this.h = Math.max(this.bottom(), other.bottom()) - this.y;
    }

    right()
    {
        return this.x + this.w;
    }

    bottom()
    {
        return this.y + this.h;
    }

    centerx()
    {
        return this.x + this.w * 0.5;
    }

    centery()
    {
        return this.y + this.h * 0.5;
    }

    intersects(other)
    {
        if(this.w == 0 || this.h == 0 || other.w == 0 || other.h == 0) return false;
        if(other.x < this.x + this.w && this.x < other.x + other.w && other.y < this.y + this.h)
            return this.y < other.y + other.h;
        else
            return false;
    }

    toString()
    {
        return `(${this.x}, ${this.y}, ${this.w}, ${this.h})`;
    }
}

/** @param {CanvasRenderingContext2D} ctx*/
function drawFrame(ctx, strip, frameIndex, x, y, flipX = false)
{
    console.assert(frameIndex < strip.frames.length, "invalid frame: "+frameIndex);
    let frame = strip.frames[frameIndex];
    ctx.save();
    let finalx = Math.floor(x - frame.pivotx);
    let finaly = Math.floor(y - frame.pivoty);
    if(flipX)
    {
        ctx.translate(finalx, finaly);
        ctx.scale(-1, 1);
        ctx.drawImage(frame.img, 0, 0);
    }
    else
    {
        ctx.translate(finalx, finaly);
        ctx.drawImage(frame.img, 0, 0);
    }
    ctx.restore();
    return frame;
}

/** @param {CanvasRenderingContext2D} ctx*/
function drawRect(ctx, r)
{
    ctx.beginPath();
    ctx.lineWidth = 1;
    ctx.rect(r.x, r.y, r.w, r.h);
    ctx.stroke();
}

/** @param {CanvasRenderingContext2D} ctx*/
function drawRectAt(ctx, x, y, w, h)
{
    ctx.beginPath();
    ctx.lineWidth = 1;
    ctx.rect(x, y, w, h);
    ctx.stroke();
}

class FrameTimer
{
    constructor()
    {
        this.elapsed = 0;
        this.looping = false;
        this.frame = 0;
        this.frameCount = 0;
        this.started = false;
        this.finished = false;
        this.fps = 1;
    }

    stop()
    {
        this.started = false;
        this.frame = 0;
        this.finished = false;
    }

    running()
    {
        return this.started && !this.finished;
    }

    once(frameCount, fps)
    {
        this.elapsed = 0;
        this.looping = false;
        this.frame = 0;
        this.frameCount = frameCount;
        this.started = true;
        this.finished = false;
        this.fps = fps;
    }

    loop(frameCount, fps)
    {
        this.elapsed = 0;
        this.looping = true;
        this.frame = 0;
        this.frameCount = frameCount;
        this.started = true;
        this.finished = false;
        this.fps = fps;
    }

    update(dt)
    {
        if(!this.started) return;
        this.elapsed += dt;
        let frameDuration = 1.0 / this.fps;
        while(this.elapsed >= frameDuration)
        {
            this.elapsed -= frameDuration;
            if(this.frame + 1 == this.frameCount)
            {
                if(this.looping)
                {
                    this.frame = 0;
                }
                else
                {
                    this.finished = true;
                }
            }
            else
            {
                this.frame += 1;
            }
        }
    }
}

function arraysEqual(a1, a2)
{
    if(a1.length != a2.length) return false;
    for(let i = 0; i < a1.length; i++)
    {
        if(a1[i] != a2[i]) return false;
    }
    return true;
}

function lerp( a, b, alpha ) 
{
    return a + alpha * ( b - a );
}

function distance(x1, y1, x2, y2)
{
    return Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

function distanceSq(x1, y1, x2, y2)
{
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

const FONT_TOP = 4;
const FONT_CENTER = 1;
const FONT_VCENTER = 2;
const FONT_RIGHT = 8;
const FONT_BOTTOM = 16;

class Glyph
{
    constructor()
    {
        this.rect = new Rect();
        this.pivotx = 0;
        this.pivoty = 0;
        this.offsetx = 0;
        this.offsety = 0;
    }
}

class BitmapFont
{
    constructor()
    {
        this.chars = null;
        this.glyphs = [];
        this.lineh = 0;
        this.max_char_height = 0;
        this.char_sep = 0;
        this.hasCapitalization = true;
        /** @type {Strip} */
        this.strip = null;
        this.spaceWidth = 0;
    }

    loadFromStrip(strip, textMap, hasCapitalization)
    {
        this.strip = strip;
        this.hasCapitalization = hasCapitalization;
        this.chars = textMap;
        this.lineh = strip.frames[0].rect.h;
        this.max_char_height = this.lineh;
        for(let i = 0; i < textMap.length; i++) 
        {
            const c = textMap[i];
            let fr = strip.frames[i];
            let g = new Glyph();
            g.rect.copyFrom(fr.rect);
            g.pivotx = fr.pivotx;
            g.pivoty = fr.pivoty;
            this.glyphs.push(g);
        }
    }

    drawLine(ctx, text, x, y, centering = 0, clip = null)
    {
        let area = new Rect();
        area.w = 10000;
        area.h = 10000;
        area.x = x;
        area.y = y;

        let startx = area.x;
        let starty = area.y;
        let size = this.processLine(ctx, text, startx, starty, area.w, false, clip);
        if((centering & FONT_CENTER) != 0)
        {
            startx = x - size.w * 0.5;
        }
        else
            if((centering & FONT_RIGHT) != 0)
            {
                startx = x - size.w;
            }

        if((centering & FONT_VCENTER) != 0)
        {
            starty = y + this.max_char_height * 0.5;
        }
        if((centering & FONT_TOP) != 0)
        {
            starty = y + this.lineh;
        }
        startx = Math.floor(startx);
        starty = Math.floor(starty);
        return this.processLine(ctx, text, startx, starty, area.w, true, clip);
    }

    paragraphSize(text, width)
    {
        let ret = [0,0];
        let lines = this.wordwrap(text, width);
        for(let curline of lines)
        {
            let rect = this.processLine(null, curline, 0, 0, width, false, null);
            ret[0] = Math.max(ret[0], rect.w);
            ret[1] += rect.h;
        }
        return ret;
    }

    drawParagraph(ctx, text, area, clip = null)
    {
        let lines = this.wordwrap(text, area.w);
        let offy = area.y + this.lineh; // move the baseline into the area
        for(let curline of lines)
        {
            this.drawLine(ctx, curline, area.x, offy, 0, clip);
            offy += this.lineh;
        }
    }

    wordwrap(text, width)
    {
        // TODO: hack
        if(!this.hasCapitalization)
        {
            text = text.toUpperCase();
        }

        let wordstart = 0;
        let linebegin = 0;
        let offx = 0;
        let lines = [];
        let i = 0;
        while(i < text.length)
        {
            let char = text.charAt(i);
            if(char == "\n")
            {
                // TODO: duplicated code
                let line = text.substring(linebegin, i).trim();
                if(line.length == 0) return ["STRING OVERFLOW"]; // fail
                lines.push(line);
                i += 1;
                linebegin = i;
                wordstart = i;
                offx = 0;
                continue;
            }

            let index = this.chars.indexOf(char);
            console.assert(index >= 0, "character "+char+" not found");
            let g = this.glyphs[index];

            offx += g.rect.w + g.offsetx + this.char_sep;
            if(char == " ")
            {
                wordstart = i + 1;
                i++;
                continue;
            }

            if(offx > width)
            {
                offx = 0;
                let line = text.substring(linebegin, wordstart - 1).trim();
                if(line.length == 0) return ["STRING OVERFLOW"]; // fail
                lines.push(line);
                linebegin = wordstart;
                i = wordstart;
            }
            else
            {
                i++;
            }
        }

        if(linebegin < i)
        {
            lines.push(text.substring(linebegin, text.length));
        }
        return lines;
    }

    /** @param ctx {CanvasRenderingContext2D} */
    processLine(ctx, text, x, y, width, render, clip)
    {
        if(!this.hasCapitalization) text = text.toUpperCase();
        let offx = 0;
        let maxW = 0;
        let startx = x;
        let starty = y;
        for (let i = 0; i < text.length; i++) 
        {
            const char = text[i];
            let index = this.chars.indexOf(char);
            console.assert(index >= 0, "invalid char: "+char + " in string:"+text);
            let g = this.glyphs[index];
            let glyphW = g.rect.w;
            if(char == " " && this.spaceWidth > 0) glyphW = this.spaceWidth; 
            if(render)
            {
                let frame = this.strip.frames[index];
                // ctx.drawImage(frame.img, 0, 0, frame.img.width, frame.img.height, startx + offx + g.offsetx - g.pivotx, starty - g.offsety - g.pivoty, frame.img.width, frame.img.height);
                ctx.drawImage(frame.img, startx + offx + g.offsetx - g.pivotx, starty - g.offsety - g.pivoty, frame.img.width, frame.img.height);
            }
            if(offx + glyphW + g.offsetx + this.char_sep > width)
            {
                break; // truncate
            }
            offx += glyphW;
            if(i < text.length - 1)
            {
                offx += g.offsetx + this.char_sep;
            }
            maxW = Math.max(maxW, offx);
        }
        let ret = new Rect();
        ret.x = x;
        ret.y = y - this.lineh;
        ret.w = maxW;
        ret.h = this.lineh;
        return ret;
    }
}

function stopSound(snd)
{
    // snd.stop();
    snd.pause(); // TODO: is this right??
    snd.currentTime = 0;
}

function pauseSound(snd)
{
    snd.pause();
    snd.currentTime = 0;
}

function playSound(snd, volume = 1, loop = false)
{
    if(!playerInteracted) return;
    if(!snd.paused) return;
    snd.volume = volume;
    snd.loop = loop;
    let promise = snd.play();
    if(promise !== undefined)
    {
        promise.then(_ =>
        {
            // Autoplay started!
        }).catch(error =>
        {
            console.log(error + " trying to play " + snd.src);
            // Autoplay was prevented.
            // Show a "Play" button so that user can start playback.
        });
    }
}

function canvasFromImage(image)
{
    let ret = document.createElement("canvas");
    ret.width = image.width;
    ret.height = image.height;
    let ctx = get2DContext(ret);
    ctx.drawImage(image, 0, 0);
    return ret;
}

function shuffle(array) 
{
    let currentIndex = array.length;
  
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
  
      // Pick a remaining element...
      let randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
}

function rnd(start, end)
{
    return Math.floor(Math.random() * (end - start) + start);
}

function r2d(r)
{
    return 180 * r / Math.PI;
}

function clamp01(v)
{
    return Math.max(0, Math.min(1, v));
}

function randomColor()
{
    return Math.floor(Math.random() * 16777215).toString(16);
}

function pickRandomArrayElement(theArray)
{
    let index = rnd(0, theArray.length);
    return theArray[index];
}

function activatePlayerInteraction()
{
    if(!playerInteracted)
    {
        // console.log("activating player interaction");
        playerInteracted = true;
        // play one sound to trigger the sound loadings
        if(allSounds.length > 0)
        {
            let snd = allSounds[0];
            let old = snd.volume;
            snd.volume = 0.00001;
            snd.play();
            snd.pause();
            snd.volume = old;
            // snd.currentTime = 0;
        }
    }
}

function tintImage(image, color)
{
    let r = color >> 16 & 0xff;
    let g = color >> 8 & 0xff;
    let b = color >> 0 & 0xff;
    let ctx = get2DContext(image);
    let imgData = ctx.getImageData(0, 0, image.width, image.height);
    for(let i = 0; i < imgData.data.length; i += 4) 
    {
        
        imgData.data[i + 0] = r;
        imgData.data[i + 1] = g;
        imgData.data[i + 2] = b;
    }
    ctx.putImageData(imgData, 0, 0);
}

function multiplyImageColor(image, colorMultiplier)
{
    let r = (colorMultiplier >> 16 & 0xff)/0xff;
    let g = (colorMultiplier >> 8 & 0xff)/0xff;
    let b = (colorMultiplier >> 0 & 0xff)/0xff;
    let ctx = image.getContext("2d");
    let imgData = ctx.getImageData(0, 0, image.width, image.height);
    for(let i = 0; i < imgData.data.length; i += 4) 
    {
        imgData.data[i + 0] *= r;
        imgData.data[i + 1] *= g;
        imgData.data[i + 2] *= b;
    }
    ctx.putImageData(imgData, 0, 0);
}

function isMobile()
{
    let isMobile = ('ontouchstart' in document.documentElement && /mobi/i.test(navigator.userAgent));
    // isMobile = true;
    return isMobile;
}

function supportsRightClick()
{
    return !isMobile();
}

/** @param ctx {CanvasRenderingContext2D} */
function showLoadingC64(ctx, rect)
{
    // c64 colors
    let colorsWeb = ["#000000", "#FFFFFF", "#880000", "#AAFFEE", "#CC44CC", "#00CC55", "#0000AA", "#EEEE77", "#DD8855", "#664400", "#FF7777", "#333333", "#777777", "#AAFF66", "#0088FF", "BBBBBB"];
    let colors = ["#000000", "#3e31a2", "#574200", "#8c3e34", "#545454", "#8d47b3", "#905f25", "#7c70da", "#808080", "#68a941", "#bb776d", "#7abfc7", "#ababab", "#d0dc71", "#acea88", "ffffff"];
    let bandH = 6;
    ctx.save();
    ctx.imageSmoothingEnabled = false;

    let bandCount = Math.floor(rect.h / bandH) + 1;
    let offy = 0;
    for(let i = 0; i < bandCount; i++) 
    {
        ctx.fillStyle = pickRandomArrayElement(colors);
        ctx.fillRect(rect.x, rect.y + offy, rect.w, bandH);
        offy += bandH;
    }
    ctx.restore();
}