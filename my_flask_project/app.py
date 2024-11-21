# Import necessary modules
import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort


def get_db_connection():
    conn = sqlite3.connect("flask_blog/database.db")
    conn.row_factory = sqlite3.Row
    return conn


def get_post(post_id):
    # open the connection to db
    conn = get_db_connection()
    # select the post base on it's id
    post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    # clos the connection
    conn.close()
    # checking if we already have the post or not
    if post is None:
        abort(404)
    return post


# Create a Flask application instance
app = Flask(__name__)
app.config["SECRET_KEY"] = "your secret key"


# Define a view function for the main route '/'
@app.route("/")
def index():
    conn = get_db_connection()
    posts = conn.execute("SELECT * FROM posts").fetchall()
    conn.close()
    return render_template("index.html", posts=posts)


@app.route("/<int:post_id>")
def post(post_id):
    post = get_post(post_id)
    return render_template("post.html", post=post)


@app.route("/create", methods=("GET", "POST"))
def create():
    # If the user clicked on Submit, it sends a POST request
    if request.method == "POST":
        # Get the title and save it in a variable
        title = request.form["title"]
        # Get the content the user wrote and save it in a variable
        content = request.form["content"]

        # Check if title is provided
        if not title:
            flash("Title is required!")
        else:
            # Open a connection to the database
            conn = get_db_connection()
            # Insert the new values in the database
            conn.execute(
                "INSERT INTO posts (title, content) VALUES (?, ?)", (title, content)
            )
            conn.commit()
            conn.close()
            # Redirect the user to the index page
            return redirect(url_for("index"))

    # Render the create page if method is GET or validation fails
    return render_template("create.html")


@app.route("/<int:id>/edit", methods=("GET", "POST"))
def edit(id):
    # Get the post to be edited by it's id
    post = get_post(id)

    if request.method == "POST":
        title = request.form["title"]
        content = request.form["content"]

        if not title:
            flash("Title is required!")
        else:
            conn = get_db_connection()
            # Update the table
            conn.execute(
                "UPDATE posts SET title = ?, content = ?" " WHERE id = ?",
                (title, content, id),
            )
            conn.commit()
            conn.close()
            return redirect(url_for("index"))

    return render_template("edit.html", post=post)


@app.route("/<int:id>/delete", methods=("POST",))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute("DELETE FROM posts WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post["title"]))
    return redirect(url_for("index"))