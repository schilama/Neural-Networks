  <p>In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:</p>

<ul>
  <li>understand the basic <strong>Image Classification pipeline</strong> and the data-driven approach (train/predict stages)</li>
  <li>understand the train/val/test <strong>splits</strong> and the use of validation data for <strong>hyperparameter tuning</strong>.</li>
  <li>develop proficiency in writing efficient <strong>vectorized</strong> code with numpy</li>
  <li>implement and apply a k-Nearest Neighbor (<strong>kNN</strong>) classifier</li>
  <li>implement and apply a Multiclass Support Vector Machine (<strong>SVM</strong>) classifier</li>
  <li>implement and apply a <strong>Softmax</strong> classifier</li>
  <li>implement and apply a <strong>Two layer neural network</strong> classifier</li>
  <li>understand the differences and tradeoffs between these classifiers</li>
  <li>get a basic understanding of performance improvements from using <strong>higher-level representations</strong> than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)</li>
</ul>

<h3 id="setup">Setup</h3>
<p>Get the code as a zip file <a href="http://vis-www.cs.umass.edu/682/asgns/assignment1.zip">here</a>. As for the dependencies:</p>

<p><strong>[Option 1] Use Anaconda:</strong>
The preferred approach for installing all the assignment dependencies is to use <a href="https://www.continuum.io/downloads">Anaconda</a>, which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis. Once you install it you can skip all mentions of requirements and you’re ready to go directly to working on the assignment.</p>

<p><strong>[Option 2] Manual install, virtual environment:</strong>
If you’d like to (instead of Anaconda) go with a more manual and risky installation route you will likely want to create a <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/">virtual environment</a> for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following:</p>

<div class="language-bash highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="nb">cd </span>assignment1
<span class="nb">sudo </span>pip install virtualenv      <span class="c"># This may already be installed</span>
virtualenv .env                  <span class="c"># Create a virtual environment</span>
<span class="nb">source</span> .env/bin/activate         <span class="c"># Activate the virtual environment</span>
pip install <span class="nt">-r</span> requirements.txt  <span class="c"># Install dependencies</span>
<span class="c"># Work on the assignment for a while ...</span>
deactivate                       <span class="c"># Exit the virtual environment</span>
</code></pre></div></div>

<p><strong>Download data:</strong>
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the <code class="highlighter-rouge">assignment1</code> directory:</p>

<div class="language-bash highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="nb">cd </span>datasets
./get_datasets.sh
</code></pre></div></div>

<p><strong>Start Jupyter Notebook:</strong>
After you have the CIFAR-10 data, you should start the Jupyter Notebook server from the
<code class="highlighter-rouge">assignment1</code> directory. If you are unfamiliar with Jupyter, you should read our
<a href="/notes/jupyter-tutorial/">Jupyter tutorial</a>.</p>

<p><strong>NOTE:</strong> If you are working in a virtual environment on OSX, you may encounter
errors with matplotlib due to the <a href="http://matplotlib.org/faq/virtualenv_faq.html">issues described here</a>. You can work around this issue by starting the Jupyter server using the <code class="highlighter-rouge">start_jupyter_osx.sh</code> script from the <code class="highlighter-rouge">assignment1</code> directory; the script assumes that your virtual environment is named <code class="highlighter-rouge">.env</code>.</p>

<h3 id="submitting-your-work">Submitting your work</h3>

<p>To make sure everything is working properly, <strong>remember to do a clean run (“Kernel -&gt; Restart &amp; Run All”) after you finish work for each notebook</strong> and submit the final version with all the outputs. 
Once you are done working, zip all the code and notebooks in a single file and upload it to Moodle. On Linux or macOS you can run the provided <code class="highlighter-rouge">collectSubmission.sh</code> script from <code class="highlighter-rouge">assignment1/</code> to produce a file <code class="highlighter-rouge">assignment1.zip</code>.</p>

<h3 id="q1-k-nearest-neighbor-classifier-20-points">Q1: k-Nearest Neighbor classifier (20 points)</h3>

<p>The Jupyter Notebook <strong>knn.ipynb</strong> will walk you through implementing the kNN classifier.</p>

<h3 id="q2-training-a-support-vector-machine-25-points">Q2: Training a Support Vector Machine (25 points)</h3>

<p>The Jupyter Notebook <strong>svm.ipynb</strong> will walk you through implementing the SVM classifier.</p>

<h3 id="q3-implement-a-softmax-classifier-20-points">Q3: Implement a Softmax classifier (20 points)</h3>

<p>The Jupyter Notebook <strong>softmax.ipynb</strong> will walk you through implementing the Softmax classifier.</p>

<h3 id="q4-two-layer-neural-network-25-points">Q4: Two-Layer Neural Network (25 points)</h3>
<p>The Jupyter Notebook <strong>two_layer_net.ipynb</strong> will walk you through the implementation of a two-layer neural network classifier.</p>

<h3 id="q5-higher-level-representations-image-features-10-points">Q5: Higher Level Representations: Image Features (10 points)</h3>

<p>The Jupyter Notebook <strong>features.ipynb</strong> will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.</p>

<h3 id="q6-cool-bonus-do-something-extra-10-points">Q6: Cool Bonus: Do something extra! (+10 points)</h3>

<p>Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or anything fun to look at? Or maybe you can experiment with a spin on the loss function? If you try out something cool we’ll give you up to 10 extra points and may feature your results in the lecture.</p>

