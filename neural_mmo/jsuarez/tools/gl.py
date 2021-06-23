# author: Somsubhra Bairi (201101056)

# Draws a polyhedron by cutting corners of cube
# Controls: UP - rotate up
#           DOWN - rotate down
#           LEFT - rotate left
#           RIGHT - rotate right

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")


# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self):
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.scale = 2.0

    # Initialize
    def init(self):
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Set the shade model to flat
        glShadeModel(GL_FLAT)

    # Draw half of the cube with corners cut
    def draw_half(self, mirror):

        # The plane equations cutting corners of cube
        eqn = [-1.0, 0.0, 0.0, 0.0]
        eqn1 = [1.0, 1.0, 1.0, 1.25]
        eqn2 = [1.0, -1.0, 1.0, 1.25]
        eqn3 = [1.0, 1.0, -1.0, 1.25]
        eqn4 = [1.0, -1.0, -1.0, 1.25]
        eqn5 = [-1.0, 1.0, 1.0, 1.25]

        # Set the color to white
        glColor3f(1.0, 1.0, 1.0)

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        if mirror:
            glScalef(-self.scale, self.scale, self.scale)
            glRotatef(-self.rotate_y, 0.0, 1.0, 0.0)
        else:
            glScalef(self.scale, self.scale, self.scale)
            glRotatef(self.rotate_y, 0.0, 1.0, 0.0)

        glRotatef(self.rotate_x, 1.0, 0.0, 0.0)

        # Draw solid cube
        glutSolidCube(1.0)

        # Draw a red wire cube to highlight the background
        glColor3f(1.0, 0.0, 0.0)
        glutWireCube(1.0)

        # Clip the corners of the cube with these equations
        glClipPlane(GL_CLIP_PLANE0, eqn1)
        glEnable(GL_CLIP_PLANE0)

        glClipPlane(GL_CLIP_PLANE1, eqn2)
        glEnable(GL_CLIP_PLANE1)

        glClipPlane(GL_CLIP_PLANE2, eqn3)
        glEnable(GL_CLIP_PLANE2)

        glClipPlane(GL_CLIP_PLANE3, eqn4)
        glEnable(GL_CLIP_PLANE3)

        glClipPlane(GL_CLIP_PLANE4, eqn5)
        glEnable(GL_CLIP_PLANE4)

        # Cut the cube into half
        glClipPlane(GL_CLIP_PLANE5, eqn)
        glEnable(GL_CLIP_PLANE5)

        glFlush()

    # The display function
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw half the cube with corners cut
        self.draw_half(False)

        # Draw a mirror image of the above half cube
        self.draw_half(True)

    # The reshape function
    def reshape(self, w, h):
        glViewport(0, 0, GLsizei(w), GLsizei(h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
        glMatrixMode(GL_MODELVIEW)

    # The keyboard controls
    def special(self, key, x, y):

        # Rotate cube according to keys pressed
        if key == GLUT_KEY_RIGHT:
            self.rotate_y += 5
        if key == GLUT_KEY_LEFT:
            self.rotate_y -= 5
        if key == GLUT_KEY_UP:
            self.rotate_x += 5
        if key == GLUT_KEY_DOWN:
            self.rotate_x -= 5
        glutPostRedisplay()


# The main function
def main():

    # Initialize OpenGL
    glutInit(sys.argv)

    # Set display mode
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

    # Set size and position of window size
    glutInitWindowSize(400, 400)
    glutInitWindowPosition(100, 100)

    # Create window with given title
    glutCreateWindow("Cube")

    # Instantiate the cube
    cube = Cube()

    cube.init()

    # The callback for display function
    glutDisplayFunc(cube.display)

    # The callback for reshape function
    glutReshapeFunc(cube.reshape)

    # The callback function for keyboard controls
    glutSpecialFunc(cube.special)

    # Start the main loop
    glutMainLoop()

# Call the main function
if __name__ == '__main__':
    main()
