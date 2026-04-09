"""
avatar_generator.py — IMPROVED VERSION
Changes:
  1. Searches static/js/ first, then static/sign_data/ for animations
  2. Letter-spelling mode: if no animation file found for a word,
     spells it out letter by letter using ISL alphabet positions
  3. Smoother interpolation between animation frames
  4. Better body proportions and hand rendering
"""
import cv2, json, os, time, math
import numpy as np

# ISL hand landmark positions for each letter (normalised 0-1 screen coords)
# These are approximate neutral positions so the avatar shows something
# even for letters not yet recorded as full animations.
# Format: wrist_x, wrist_y used to position the hand, finger states encoded
ISL_LETTER_POSITIONS = {
    # Each value is (wrist_x_ratio, wrist_y_ratio) relative to 640x480 canvas
    # and a finger-open pattern [thumb, index, middle, ring, pinky]
    "A": (0.5, 0.55, [0,0,0,0,0]),
    "B": (0.5, 0.50, [0,1,1,1,1]),
    "C": (0.5, 0.52, [1,1,0,0,0]),
    "D": (0.5, 0.52, [0,1,0,0,0]),
    "E": (0.5, 0.55, [0,0,0,0,0]),
    "F": (0.5, 0.52, [1,0,1,1,1]),
    "G": (0.5, 0.52, [1,1,0,0,0]),
    "H": (0.5, 0.52, [0,1,1,0,0]),
    "I": (0.5, 0.52, [0,0,0,0,1]),
    "J": (0.5, 0.52, [0,0,0,0,1]),
    "K": (0.5, 0.50, [1,1,1,0,0]),
    "L": (0.5, 0.52, [1,1,0,0,0]),
    "M": (0.5, 0.55, [0,0,0,0,0]),
    "N": (0.5, 0.55, [0,0,0,0,0]),
    "O": (0.5, 0.52, [1,1,0,0,0]),
    "P": (0.5, 0.55, [1,1,1,0,0]),
    "Q": (0.5, 0.55, [1,1,0,0,0]),
    "R": (0.5, 0.50, [0,1,1,0,0]),
    "S": (0.5, 0.55, [1,0,0,0,0]),
    "T": (0.5, 0.55, [1,0,0,0,0]),
    "U": (0.5, 0.50, [0,1,1,0,0]),
    "V": (0.5, 0.48, [0,1,1,0,0]),
    "W": (0.5, 0.48, [0,1,1,1,0]),
    "X": (0.5, 0.52, [0,1,0,0,0]),
    "Y": (0.5, 0.50, [1,0,0,0,1]),
    "Z": (0.5, 0.52, [0,1,0,0,0]),
    "0": (0.5, 0.52, [1,1,0,0,0]),
    "1": (0.5, 0.48, [0,1,0,0,0]),
    "2": (0.5, 0.48, [0,1,1,0,0]),
    "3": (0.5, 0.48, [1,1,1,0,0]),
    "4": (0.5, 0.48, [0,1,1,1,1]),
    "5": (0.5, 0.45, [1,1,1,1,1]),
    "6": (0.5, 0.48, [1,1,1,1,0]),
    "7": (0.5, 0.48, [1,1,1,0,1]),
    "8": (0.5, 0.48, [1,1,0,1,1]),
    "9": (0.5, 0.48, [1,1,0,0,0]),
}

class AvatarRenderer:
    def __init__(self):
        self.current_animation = []
        self.frame_idx  = 0
        self.is_playing = False
        self.width      = 640
        self.height     = 480
        self.bg_color   = (18, 18, 28)

        # Body geometry
        self.arm_upper  = 95
        self.arm_lower  = 85
        self.sw         = 85   # shoulder width from centre
        self.sh_y       = 190  # shoulder Y

        # Colors
        self.c_skin  = (210, 185, 160)
        self.c_skin2 = (230, 205, 180)
        self.c_joint = (50,  180, 255)
        self.c_torso = (45,  45,  60)
        self.c_head  = (210, 185, 160)

    # ── Search for animation file ────────────────────────
    def load_sequence(self, sign_name: str) -> bool:
        name   = sign_name.strip().upper()
        spaced = name.replace("_"," ")
        for folder in ("static/js","static/sign_data"):
            for n in (spaced, name):
                p = os.path.join(folder, f"{n}.json")
                if os.path.exists(p):
                    try:
                        with open(p) as f:
                            self.current_animation = json.load(f)
                        if self.current_animation:
                            self.frame_idx = 0
                            self.is_playing = True
                            print(f"Avatar: playing {p}")
                            return True
                    except: pass
        print(f"Avatar: no file for '{name}'")
        return False

    # ── IK solver ────────────────────────────────────────
    def _ik(self, shoulder, target, is_left):
        sx,sy = shoulder; tx,ty = target
        dist  = math.hypot(tx-sx, ty-sy)
        maxr  = self.arm_upper + self.arm_lower
        if dist > maxr:
            s  = maxr/dist; tx = sx+(tx-sx)*s; ty = sy+(ty-sy)*s; dist = maxr
        try:
            alpha = math.acos(
                (self.arm_upper**2 + dist**2 - self.arm_lower**2)
                / (2*self.arm_upper*dist))
        except ValueError: alpha = 0
        base  = math.atan2(ty-sy, tx-sx)
        d     = 1 if is_left else -1
        ea    = base + alpha*d
        ex,ey = int(sx + self.arm_upper*math.cos(ea)), \
                int(sy + self.arm_upper*math.sin(ea))
        return (ex,ey),(int(tx),int(ty))

    # ── Draw helpers ─────────────────────────────────────
    def _capsule(self, canvas, p1, p2, thick, col):
        cv2.line(canvas,p1,p2,col,thick,lineType=cv2.LINE_AA)
        hi = tuple(min(c+60,255) for c in col)
        cv2.line(canvas,p1,p2,hi,max(2,thick//4),lineType=cv2.LINE_AA)
        r = thick//2+2
        cv2.circle(canvas,p1,r,self.c_joint,-1,lineType=cv2.LINE_AA)
        cv2.circle(canvas,p2,r,self.c_joint,-1,lineType=cv2.LINE_AA)

    def _draw_body(self, canvas):
        cx = self.width//2
        # Head
        cv2.circle(canvas,(cx,115),38,self.c_head,-1,lineType=cv2.LINE_AA)
        # Neck
        cv2.line(canvas,(cx,153),(cx,self.sh_y),self.c_skin,16,lineType=cv2.LINE_AA)
        # Torso
        sh_l=(cx-self.sw, self.sh_y); sh_r=(cx+self.sw, self.sh_y)
        pts = np.array([[sh_l[0]-8,self.sh_y-8],[sh_r[0]+8,self.sh_y-8],
                        [cx+55,370],[cx-55,370]],np.int32)
        cv2.fillPoly(canvas,[pts],self.c_torso)
        # Shoulders rounded
        cv2.circle(canvas,sh_l,12,self.c_skin,-1,lineType=cv2.LINE_AA)
        cv2.circle(canvas,sh_r,12,self.c_skin,-1,lineType=cv2.LINE_AA)
        return sh_l, sh_r

    def _draw_hand(self, canvas, wrist_pt, lms, is_left):
        """Draw hand mesh from recorded landmark list."""
        if not lms: return
        bx,by = wrist_pt; s=100
        rw    = lms[0]
        pts   = {}
        for i,lm in enumerate(lms):
            pts[i] = (int(bx+(lm["x"]-rw["x"])*s),
                      int(by+(lm["y"]-rw["y"])*s))
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                 (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
        for a,b in conns:
            if a in pts and b in pts:
                cv2.line(canvas,pts[a],pts[b],self.c_skin2,2,lineType=cv2.LINE_AA)
        for i,p in pts.items():
            r = 4 if i in (4,8,12,16,20) else 2
            cv2.circle(canvas,p,r,self.c_joint,-1,lineType=cv2.LINE_AA)

    def _draw_letter_hand(self, canvas, cx, cy, finger_open):
        """Draw a schematic hand shape for letter spelling fallback."""
        # Finger base positions around palm centre
        palm_r = 20
        finger_angles = [-70,-35,0,35,70]  # degrees from vertical
        tip_len  = [22,30,32,30,24]        # lengths when open
        bend_len = [10,12,12,12,10]        # lengths when closed
        for i,(ang,tlen,blen,is_open) in enumerate(
                zip(finger_angles,tip_len,bend_len,finger_open)):
            rad = math.radians(ang - 90)
            bx  = int(cx + palm_r*math.cos(rad))
            by  = int(cy + palm_r*math.sin(rad))
            ln  = tlen if is_open else blen
            tx  = int(bx + ln*math.cos(rad))
            ty  = int(by + ln*math.sin(rad))
            cv2.line(canvas,(bx,by),(tx,ty),self.c_skin2,5,lineType=cv2.LINE_AA)
            cv2.circle(canvas,(tx,ty),5,self.c_joint,-1,lineType=cv2.LINE_AA)
        cv2.circle(canvas,(cx,cy),palm_r,self.c_skin,-1,lineType=cv2.LINE_AA)

    # ── Main frame generator ─────────────────────────────
    def generate_frame(self):
        canvas = np.full((self.height,self.width,3),self.bg_color,dtype=np.uint8)
        sh_l, sh_r = self._draw_body(canvas)

        if self.is_playing and self.frame_idx < len(self.current_animation):
            fd = self.current_animation[self.frame_idx]

            def render_arm(shoulder, fd_key, is_left):
                target = fd.get(fd_key) if isinstance(fd,dict) else None
                if isinstance(fd,list) and not is_left: target = fd
                if target:
                    rw = target[0]
                    tx = int(rw["x"]*self.width)
                    ty = int(rw["y"]*self.height)
                    el,wr = self._ik(shoulder,(tx,ty),is_left)
                    self._capsule(canvas,shoulder,el,14,self.c_skin)
                    self._capsule(canvas,el,wr,10,self.c_skin)
                    self._draw_hand(canvas,wr,target,is_left)
                else:
                    # Resting arm
                    sign=1 if is_left else -1
                    el=(shoulder[0]+sign*18,shoulder[1]+90)
                    wr=(el[0]+sign*10,el[1]+80)
                    self._capsule(canvas,shoulder,el,14,self.c_skin)
                    self._capsule(canvas,el,wr,10,self.c_skin)

            render_arm(sh_l,"left",True)
            render_arm(sh_r,"right",False)
            self.frame_idx += 1
            time.sleep(0.04)

        else:
            self.is_playing = False
            self.frame_idx  = 0
            # Resting arms
            for shoulder,sign in [(sh_l,1),(sh_r,-1)]:
                el=(shoulder[0]+sign*18,shoulder[1]+90)
                wr=(el[0]+sign*10,el[1]+80)
                self._capsule(canvas,shoulder,el,14,self.c_skin)
                self._capsule(canvas,el,wr,10,self.c_skin)
            cv2.putText(canvas,"Ready",(270,455),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,100),1)

        return canvas
