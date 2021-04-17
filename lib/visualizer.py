# python 3.7
"""Utility functions for visualizing results."""

import base64
import os.path
import cv2, math
import numpy as np
from bs4 import BeautifulSoup
import matplotlib
import matplotlib.style as style
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

from lib.op import bu


__all__ = [
    'get_label_color', 'get_grid_shape', 'get_blank_image',
    'load_image', 'save_image',
    'resize_image', 'add_text_to_image', 'parse_image_size', 'fuse_images',
    'segviz_numpy', 'segviz_torch', 'heatmap_numpy', 'maskviz',
    'HtmlPageVisualizer', 'HtmlPageReader', 'VideoReader', 'VideoWriter'
]


def plot_dict(dic, fpath=None, N_row=None, N_col=None):
  if N_row is None:
    N = len(dic.keys())
    N_row = math.ceil(math.sqrt(N))
    N_col = math.ceil(N / N_row)
  
  fig = plt.figure(figsize=(6 * N_col, 4 * N_row))
  for i, (k, v) in enumerate(dic.items()):
    ax = plt.subplot(N_row, N_col, i + 1)
    if type(v) is dict: # multiple lines with legend
      for iv in v.values():
        ax.plot(iv)
      ax.legend(list(v.keys()))
    else:
      ax.plot(v)
    ax.set_title(k)
  plt.tight_layout()
  
  if fpath:
    plt.savefig(fpath)
    plt.close()


def get_images_SE(G, SE, P, z, size=256):
  P.eval()
  with torch.no_grad():
    if hasattr(G, "mapping"):
      wp = G.truncation(G.mapping(z))
      image, feature = G.synthesis(wp, generate_feature=True)
    else:
      image, feature = G(z, generate_feature=True)
    label = P(image, size=size).long()
  seg = SE(feature)
  return bu(image, size), seg, label


def viz_SE(G, SE, P, z, size=256):
  """Get the images, segmentations, and layer semantics for visualization."""
  images = []
  layer_vizs = []
  seg_vizs = []
  label_vizs = []
  for i in range(z.shape[0]): # batch
    image, segs, label = get_images_SE(G, SE, P, z[i:i+1])
    images.append(image)
    imgs = []
    for seg in segs:
      seg_label = bu(seg, size)[0].argmax(0)
      imgs.append(segviz_torch(seg_label))
    seg_vizs.append(imgs[-1])
    layer_vizs.append(torch.stack(imgs))
    label_vizs.append(segviz_torch(label[0]))
  images = (torch.cat(images).clamp(-1, 1).cpu() + 1) / 2
  layer_vizs = torch.stack(layer_vizs).cpu()
  seg_vizs = torch.stack(seg_vizs).cpu()
  label_vizs = torch.stack(label_vizs).cpu()
  return images, seg_vizs, label_vizs, layer_vizs


def get_label_color(label_idx):
  return high_contrast_arr[label_idx]


POSITIVE_COLOR = cm.get_cmap("Reds")
NEGATIVE_COLOR = cm.get_cmap("Blues")
def heatmap_numpy(image):
  """Get the heatmap of the image

  Args:
    image : A numpy array of shape (N, H, W) and scale in [-1, 1]

  Returns:
    A image of shape (N, H, W, 3) in [0, 1] scale
  """
  image1 = image.copy()
  mask1 = image1 > 0
  image1[~mask1] = 0

  image2 = -image.copy()
  mask2 = image2 > 0
  image2[~mask2] = 0

  pos_img = POSITIVE_COLOR(image1)[:, :, :, :3]
  neg_img = NEGATIVE_COLOR(image2)[:, :, :, :3]

  x = np.ones_like(pos_img)
  x[mask1] = pos_img[mask1]
  x[mask2] = neg_img[mask2]

  return x


def make_grid_numpy(image, num_cols=-1, padding=5, padding_value=0):
  """
  Making a grid out of a batch of images.
  Args:
    image: (N, H, W, C)
    num_cols: specify the column number. -1 for a square.
  Returns:
    An image of size (H * a, W * b, C)
  """
  N = len(image)
  H, W, C = image[0].shape
  nH, nW = H + padding, W + padding
  if num_cols < 0:
    num_cols = int(np.sqrt(N))
  num_rows = math.ceil(N / float(num_cols))
  img = np.zeros((nH * num_rows - padding, nW * num_cols - padding, 3))
  img.fill(padding_value)
  for i in range(N):
    x, y = i // num_cols, i % num_cols
    img[x * nH : x * nH + H, y * nW : y * nW + W] = image[i]
  return img


def segviz_numpy(seg):
  """Visualize the segmentation mask

  Args:
    seg : The segmentation map in (H, W), (1, H, W) or (1, 1, H, W)
  Returns:
    A image of (H, W, 3) in scale [0, 255]
  """
  while seg.shape[0] == 1:
    seg = seg.squeeze(0)
  result = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
  labels = np.unique(seg)
  for label in labels:
    if label == 0:
      continue
    bitmap = (seg == label)
    result[bitmap] = high_contrast_arr[label % len(high_contrast_arr)]
  return result


def segviz_torch(seg):
  """Pytorch tensor version of segviz_numpy"""
  x = torch.from_numpy(segviz_numpy(seg.detach().cpu().numpy()))
  return x.float().permute(2, 0, 1) / 255.


def maskviz(mask):
  """Visualize a binary mask
  Args:
    mask: (N, H, W) or (N, 1, H, W) of torch.Tensor in [0, 1]
  Returns:
    RGB format mask in a list of numpy array
  """
  if mask.shape[1] == 1:
    mask = mask.squeeze(1)
  v = torch.stack([mask * 255] * 3, 3)
  v = v.cpu().numpy().astype("uint8")
  return [i for i in v]


def get_grid_shape(size, row=0, col=0, is_portrait=False):
  """Gets the shape of a grid based on the size.

  This function makes greatest effort on making the output grid square if
  neither `row` nor `col` is set. If `is_portrait` is set as `False`, the height
  will always be equal to or smaller than the width. For example, if input
  `size = 16`, output shape will be `(4, 4)`; if input `size = 15`, output shape
  will be (3, 5). Otherwise, the height will always be equal to or larger than
  the width.

  Args:
    size: Size (height * width) of the target grid.
    is_portrait: Whether to return a portrait size of a landscape size.
      (default: False)

  Returns:
    A two-element tuple, representing height and width respectively.
  """
  assert isinstance(size, int)
  assert isinstance(row, int)
  assert isinstance(col, int)
  if size == 0:
    return (0, 0)

  if row > 0 and col > 0 and row * col != size:
    row = 0
    col = 0

  if row > 0 and size % row == 0:
    return (row, size // row)
  if col > 0 and size % col == 0:
    return (size // col, col)

  row = int(np.sqrt(size))
  while row > 0:
    if size % row == 0:
      col = size // row
      break
    row = row - 1

  return (col, row) if is_portrait else (row, col)


def get_blank_image(height, width, channels=3, is_black=True):
  """Gets a blank image, either white of black.

  NOTE: This function will always return an image with `RGB` channel order for
  color image and pixel range [0, 255].

  Args:
    height: Height of the returned image.
    width: Width of the returned image.
    channels: Number of channels. (default: 3)
    is_black: Whether to return a black image or white image. (default: True)
  """
  shape = (height, width, channels)
  if is_black:
    return np.zeros(shape, dtype=np.uint8)
  return np.ones(shape, dtype=np.uint8) * 255


def load_image(path, image_channels=3):
  """Loads an image from disk.

  NOTE: This function will always return an image with `RGB` channel order for
  color image and pixel range [0, 255].

  Args:
    path: Path to load the image from.
    image_channels: Number of image channels of returned image. This field is
      employed since `cv2.imread()` will always return a 3-channel image, even
      for grayscale image.

  Returns:
    An image with dtype `np.ndarray` or `None` if input `path` does not exist.
  """
  if not os.path.isfile(path):
    return None

  assert image_channels in [1, 3]

  image = cv2.imread(path)
  assert image.ndim == 3 and image.shape[2] == 3
  if image_channels == 1:
    return image[:, :, 0:1]
  return image[:, :, ::-1]


def save_image(path, image):
  """Saves an image to disk.

  NOTE: The input image (if colorful) is assumed to be with `RGB` channel order
  and pixel range [0, 255].

  Args:
    path: Path to save the image to.
    image: Image to save.
  """
  if image is None:
    return

  assert image.ndim == 3 and image.shape[2] in [1, 3]
  cv2.imwrite(path, image[:, :, ::-1])


def resize_image(image, *args, **kwargs):
  """Resizes image.

  This is a wrap of `cv2.resize()`.

  NOTE: THe channel order of the input image will not be changed.

  Args:
    image: Image to resize.
  """
  if image is None:
    return None

  assert image.ndim == 3 and image.shape[2] in [1, 3]
  image = cv2.resize(image, *args, **kwargs)
  if image.ndim == 2:
    return image[:, :, np.newaxis]
  return image


def add_text_to_image(image,
                      text='',
                      position=None,
                      font=cv2.FONT_HERSHEY_TRIPLEX,
                      font_size=1.0,
                      line_type=cv2.LINE_8,
                      line_width=1,
                      color=(255, 255, 255)):
  """Overlays text on given image.

  NOTE: The input image is assumed to be with `RGB` channel order.

  Args:
    image: The image to overlay text on.
    text: Text content to overlay on the image. (default: '')
    position: Target position (bottom-left corner) to add text. If not set,
      center of the image will be used by default. (default: None)
    font: Font of the text added. (default: cv2.FONT_HERSHEY_TRIPLEX)
    font_size: Font size of the text added. (default: 1.0)
    line_type: Line type used to depict the text. (default: cv2.LINE_8)
    line_width: Line width used to depict the text. (default: 1)
    color: Color of the text added in `RGB` channel order. (default:
      (255, 255, 255))

  Returns:
    An image with target text overlayed on.
  """
  if image is None or not text:
    return image

  cv2.putText(img=image,
              text=text,
              org=position,
              fontFace=font,
              fontScale=font_size,
              color=color,
              thickness=line_width,
              lineType=line_type,
              bottomLeftOrigin=False)

  return image


def parse_image_size(obj):
  """Parses object to a pair of image size, i.e., (width, height).

  Args:
    obj: The input object to parse image size from.

  Returns:
    A two-element tuple, indicating image width and height respectively.

  Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
  """
  if obj is None or obj == '':
    width = height = 0
  elif isinstance(obj, int):
    width = height = obj
  elif isinstance(obj, (list, tuple, np.ndarray)):
    numbers = tuple(obj)
    if len(numbers) == 0:
      width = height = 0
    elif len(numbers) == 1:
      width = height = numbers[0]
    elif len(numbers) == 2:
      width = numbers[0]
      height = numbers[1]
    else:
      raise ValueError(f'At most two elements for image size.')
  elif isinstance(obj, str):
    splits = obj.replace(' ', '').split(',')
    numbers = tuple(map(int, splits))
    if len(numbers) == 0:
      width = height = 0
    elif len(numbers) == 1:
      width = height = numbers[0]
    elif len(numbers) == 2:
      width = numbers[0]
      height = numbers[1]
    else:
      raise ValueError(f'At most two elements for image size.')
  else:
    raise ValueError(f'Invalid type of input: {type(obj)}!')

  return (max(0, width), max(0, height))


def fuse_images(images,
                image_size=None,
                row=0,
                col=0,
                is_row_major=True,
                is_portrait=False,
                row_spacing=0,
                col_spacing=0,
                border_left=0,
                border_right=0,
                border_top=0,
                border_bottom=0,
                black_background=True):
  """Fuses a collection of images into an entire image.

  Args:
    images: A collection of images to fuse. Should be with shape [num, height,
      width, channels].
    image_size: This field is used to resize the image before fusion. `0`
      disables resizing. (default: None)
    row: Number of rows used for image fusion. If not set, this field will be
      automatically assigned based on `col` and total number of images.
      (default: None)
    col: Number of columns used for image fusion. If not set, this field will be
      automatically assigned based on `row` and total number of images.
      (default: None)
    is_row_major: Whether the input images should be arranged row-major or
      column-major. (default: True)
    is_portrait: Only active when both `row` and `col` should be assigned
      automatically. (default: False)
    row_spacing: Space between rows. (default: 0)
    col_spacing: Space between columns. (default: 0)
    border_left: Width of left border. (default: 0)
    border_right: Width of right border. (default: 0)
    border_top: Width of top border. (default: 0)
    border_bottom: Width of bottom border. (default: 0)

  Returns:
    The fused image.

  Raises:
    ValueError: If the input `images` is not with shape [num, height, width,
      width].
  """
  if images is None:
    return images

  if images.ndim != 4:
    raise ValueError(f'Input `images` should be with shape [num, height, '
                     f'width, channels], but {images.shape} is received!')

  num, image_height, image_width, channels = images.shape
  width, height = parse_image_size(image_size)
  height = height or image_height
  width = width or image_width
  row, col = get_grid_shape(num, row=row, col=col, is_portrait=is_portrait)
  fused_height = (
      height * row + row_spacing * (row - 1) + border_top + border_bottom)
  fused_width = (
      width * col + col_spacing * (col - 1) + border_left + border_right)
  fused_image = get_blank_image(
      fused_height, fused_width, channels=channels, is_black=black_background)
  images = images.reshape(row, col, image_height, image_width, channels)
  if not is_row_major:
    images = images.transpose(1, 0, 2, 3, 4)

  for i in range(row):
    y = border_top + i * (height + row_spacing)
    for j in range(col):
      x = border_left + j * (width + col_spacing)
      if height != image_height or width != image_width:
        image = cv2.resize(images[i, j], (width, height))
      else:
        image = images[i, j]
      fused_image[y:y + height, x:x + width] = image

  return fused_image


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
  """Gets header for sortable html page.

  Basically, the html page contains a sortable table, where user can sort the
  rows by a particular column by clicking the column head.

  Example:

  column_name_list = [name_1, name_2, name_3]
  header = get_sortable_html_header(column_name_list)
  footer = get_sortable_html_footer()
  sortable_table = ...
  html_page = header + sortable_table + footer

  Args:
    column_name_list: List of column header names.
    sort_by_ascending: Default sorting order. If set as `True`, the html page
      will be sorted by ascending order when the header is clicked for the first
      time.

  Returns:
    A string, which represents for the header for a sortable html page.
  """
  header = '\n'.join([
      '<script type="text/javascript">',
      'var column_idx;',
      'var sort_by_ascending = ' + str(sort_by_ascending).lower() + ';',
      '',
      'function sorting(tbody, column_idx){',
      '  this.column_idx = column_idx;',
      '  Array.from(tbody.rows)',
      '       .sort(compareCells)',
      '       .forEach(function(row) { tbody.appendChild(row); })',
      '  sort_by_ascending = !sort_by_ascending;',
      '}',
      '',
      'function compareCells(row_a, row_b) {',
      '  var val_a = row_a.cells[column_idx].innerText;',
      '  var val_b = row_b.cells[column_idx].innerText;',
      '  var flag = sort_by_ascending ? 1 : -1;',
      '  return flag * (val_a > val_b ? 1 : -1);',
      '}',
      '</script>',
      '',
      '<html>',
      '',
      '<head>',
      '<style>',
      '  table {',
      '    border-spacing: 0;',
      '    border: 1px solid black;',
      '  }',
      '  th {',
      '    cursor: pointer;',
      '  }',
      '  th, td {',
      '    text-align: left;',
      '    vertical-align: middle;',
      '    border-collapse: collapse;',
      '    border: 0.5px solid black;',
      '    padding: 8px;',
      '  }',
      '  tr:nth-child(even) {',
      '    background-color: #d2d2d2;',
      '  }',
      '</style>',
      '</head>',
      '',
      '<body>',
      '',
      '<table>',
      '<thead>',
      '<tr>',
      ''])
  for idx, column_name in enumerate(column_name_list):
    header += f'  <th onclick="sorting(tbody, {idx})">{column_name}</th>\n'
  header += '</tr>\n'
  header += '</thead>\n'
  header += '<tbody id="tbody">\n'

  return header


def get_sortable_html_footer():
  """Gets footer for sortable html page.

  Check function `get_sortable_html_header()` for more details.
  """
  return '</tbody>\n</table>\n\n</body>\n</html>\n'


def encode_image_to_html_str(image, image_size=None):
  """Encodes an image to html language.

  NOTE: Input image is always assumed to be with `RGB` channel order.

  Args:
    image: The input image to encode. Should be with `RGB` channel order.
    image_size: This field is used to resize the image before encoding. `0`
      disables resizing. (default: None)

  Returns:
    A string which represents the encoded image.
  """
  if image is None:
    return ''

  assert image.ndim == 3 and image.shape[2] in [1, 3]

  # Change channel order to `BGR`, which is opencv-friendly.
  image = image[:, :, ::-1]

  # Resize the image if needed.
  width, height = parse_image_size(image_size)
  if height or width:
    height = height or image.shape[0]
    width = width or image.shape[1]
    image = cv2.resize(image, (width, height))

  # Encode the image to html-format string.
  encoded_image = cv2.imencode('.jpg', image)[1].tostring()
  encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
  html_str = f'<img src="data:image/jpeg;base64, {encoded_image_base64}"/>'

  return html_str


def decode_html_str_to_image(html_str, image_size=None):
  """Decodes image from html.

  Args:
    html_str: Image string parsed from html.
    image_size: This field is used to resize the image after decoding. `0`
      disables resizing. (default: None)

  Returns:
    An image with `RGB` channel order.
  """
  if not html_str:
    return None

  assert isinstance(html_str, str)
  image_str = html_str.split(',')[-1]
  encoded_image = base64.b64decode(image_str)
  encoded_image_numpy = np.frombuffer(encoded_image, dtype=np.uint8)
  image = cv2.imdecode(encoded_image_numpy, flags=cv2.IMREAD_COLOR)

  # Resize the image if needed.
  width, height = parse_image_size(image_size)
  if height or width:
    height = height or image.shape[0]
    width = width or image.shape[1]
    image = cv2.resize(image, (width, height))

  return image[:, :, ::-1]


class HtmlPageVisualizer(object):
  """Defines the html page visualizer.

  This class can be used to visualize image results as html page. Basically, it
  is based on an html-format sorted table with helper functions
  `get_sortable_html_header()`, `get_sortable_html_footer()`, and
  `encode_image_to_html_str()`. To simplify the usage, specifying the following
  fields is enough to create a visualization page:

  (1) num_rows: Number of rows of the table (header-row exclusive).
  (2) num_cols: Number of columns of the table.
  (3) header contents (optional): Title of each column.

  NOTE: `grid_size` can be used to assign `num_rows` and `num_cols`
  automatically.

  Example:

  html = HtmlPageVisualizer(num_rows, num_cols)
  html.set_headers([...])
  for i in range(num_rows):
    for j in range(num_cols):
      html.set_cell(i, j, text=..., image=..., highlight=False)
  html.save('visualize.html')
  """

  def __init__(self,
               num_rows=0,
               num_cols=0,
               grid_size=0,
               is_portrait=False,
               viz_size=None):
    if grid_size > 0:
      num_rows, num_cols = get_grid_shape(
          grid_size, row=num_rows, col=num_cols, is_portrait=is_portrait)
    assert num_rows > 0 and num_cols > 0

    self.num_rows = num_rows
    self.num_cols = num_cols
    self.viz_size = parse_image_size(viz_size)
    self.headers = ['' for _ in range(self.num_cols)]
    self.cells = [[{
        'text': '',
        'image': '',
        'highlight': False,
    } for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def set_header(self, col_idx, content):
    """Sets the content of a particular header by column index."""
    self.headers[col_idx] = content

  def set_headers(self, contents):
    """Sets the contents of all headers."""
    if isinstance(contents, str):
      contents = [contents]
    assert isinstance(contents, (list, tuple))
    assert len(contents) == self.num_cols
    for col_idx, content in enumerate(contents):
      self.set_header(col_idx, content)

  def set_cell(self, row_idx, col_idx, text='', image=None, highlight=False):
    """Sets the content of a particular cell.

    Basically, a cell contains some text as well as an image. Both text and
    image can be empty.

    Args:
      row_idx: Row index of the cell to edit.
      col_idx: Column index of the cell to edit.
      text: Text to add into the target cell. (default: None)
      image: Image to show in the target cell. Should be with `RGB` channel
        order. (default: None)
      highlight: Whether to highlight this cell in the html. (default: False)
    """
    self.cells[row_idx][col_idx]['text'] = text
    self.cells[row_idx][col_idx]['image'] = encode_image_to_html_str(
        image, self.viz_size)
    self.cells[row_idx][col_idx]['highlight'] = bool(highlight)

  def sequential_set_cell(self, images):
    """
    Sequentially assign the image to the cell
    """
    for i in range(len(images)):
      x, y = i // self.num_cols, i % self.num_cols
      self.set_cell(x, y, image=images[i])

  def save(self, save_path):
    """Saves the html page."""
    html = ''
    for i in range(self.num_rows):
      html += f'<tr>\n'
      for j in range(self.num_cols):
        text = self.cells[i][j]['text']
        image = self.cells[i][j]['image']
        color = ' bgcolor="#FF8888"' if self.cells[i][j]['highlight'] else ''
        if text:
          html += f'  <td{color}>{text}<br><br>{image}</td>\n'
        else:
          html += f'  <td{color}>{image}</td>\n'
      html += f'</tr>\n'

    header = get_sortable_html_header(self.headers)
    footer = get_sortable_html_footer()

    with open(save_path, 'w') as f:
      f.write(header + html + footer)


class HtmlPageReader(object):
  """Defines the html page reader.

  This class can be used to parse results from the visualization page generated
  by `HtmlPageVisualizer`.

  Example:

  html = HtmlPageReader(html_path)
  for j in range(html.num_cols):
    header = html.get_header(j)
  for i in range(html.num_rows):
    for j in range(html.num_cols):
      text = html.get_text(i, j)
      image = html.get_image(i, j, image_size=None)
  """
  def __init__(self, html_path):
    """Initializes by loading the content from file."""
    self.html_path = html_path
    if not os.path.isfile(html_path):
      raise ValueError(f'File `{html_path}` does not exist!')

    # Load content.
    with open(html_path, 'r') as f:
      self.html = BeautifulSoup(f, 'html.parser')

    # Parse headers.
    thead = self.html.find('thead')
    headers = thead.findAll('th')
    self.headers = []
    for header in headers:
      self.headers.append(header.text)
    self.num_cols = len(self.headers)

    # Parse cells.
    tbody = self.html.find('tbody')
    rows = tbody.findAll('tr')
    self.cells = []
    for row in rows:
      cells = row.findAll('td')
      self.cells.append([])
      for cell in cells:
        self.cells[-1].append({
            'text': cell.text,
            'image': cell.find('img')['src'],
        })
      assert len(self.cells[-1]) == self.num_cols
    self.num_rows = len(self.cells)

  def get_header(self, j):
    """Gets header for a particular column."""
    return self.headers[j]

  def get_text(self, i, j):
    """Gets text from a particular cell."""
    return self.cells[i][j]['text']

  def get_image(self, i, j, image_size=None):
    """Gets image from a particular cell."""
    return decode_html_str_to_image(self.cells[i][j]['image'], image_size)


class VideoReader(object):
  """Defines the video reader.

  This class can be used to read frames from a given video.
  """

  def __init__(self, path):
    """Initializes the video reader by loading the video from disk."""
    if not os.path.isfile(path):
      raise ValueError(f'Video `{path}` does not exist!')

    self.path = path
    self.video = cv2.VideoCapture(path)
    assert self.video.isOpened()
    self.position = 0

    self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.fps = self.video.get(cv2.CAP_PROP_FPS)

  def __del__(self):
    """Releases the opened video."""
    self.video.release()

  def read(self, position=None):
    """Reads a certain frame.

    NOTE: The returned frame is assumed to be with `RGB` channel order.

    Args:
      position: Optional. If set, the reader will read frames from the exact
        position. Otherwise, the reader will read next frames. (default: None)
    """
    if position is not None and position < self.length:
      self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
      self.position = position

    success, frame = self.video.read()
    self.position = self.position + 1

    return frame[:, :, ::-1] if success else None


class VideoWriter(object):
  """Defines the video writer.

  This class can be used to create a video.

  NOTE: `.avi` and `DIVX` is the most recommended codec format since it does not
  rely on other dependencies.
  """

  def __init__(self, path, frame_height, frame_width, fps=24, codec='DIVX'):
    """Creates the video writer."""
    self.path = path
    self.frame_height = frame_height
    self.frame_width = frame_width
    self.fps = fps
    self.codec = codec

    self.video = cv2.VideoWriter(filename=path,
                                 fourcc=cv2.VideoWriter_fourcc(*codec),
                                 fps=fps,
                                 frameSize=(frame_width, frame_height))

  def __del__(self):
    """Releases the opened video."""
    self.video.release()

  def write(self, frame):
    """Writes a target frame.

    NOTE: The input frame is assumed to be with `RGB` channel order.
    """
    self.video.write(frame[:, :, ::-1])


# A palette that maximizes perceptual contrast between entries.
# https://stackoverflow.com/questions/33295120
high_contrast = [
    [0, 0, 0], [255, 255, 0], [28, 230, 255], [255, 52, 255],
    [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89],
    [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172],
    [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135],
    [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0],
    [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128],
    [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160],
    [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0],
    [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153],
    [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111],
    [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191],
    [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9],
    [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255],
    [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102],
    [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160],
    [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76],
    [52, 54, 45], [180, 168, 189], [0, 166, 170], [69, 44, 44],
    [99, 99, 117], [163, 200, 201], [255, 145, 63], [147, 138, 129],
    [87, 83, 41], [0, 254, 207], [176, 91, 111], [140, 208, 255],
    [59, 151, 0], [4, 247, 87], [200, 161, 161], [30, 110, 0],
    [121, 0, 215], [167, 117, 0], [99, 103, 169], [160, 88, 55],
    [107, 0, 44], [119, 38, 0], [215, 144, 255], [155, 151, 0],
    [84, 158, 121], [255, 246, 159], [32, 22, 37], [114, 65, 143],
    [188, 35, 255], [153, 173, 192], [58, 36, 101], [146, 35, 41],
    [91, 69, 52], [253, 232, 220], [64, 78, 85], [0, 137, 163],
    [203, 126, 152], [164, 232, 4], [50, 78, 114], [106, 58, 76],
    [131, 171, 88], [0, 28, 30], [209, 247, 206], [0, 75, 40],
    [200, 208, 246], [163, 164, 137], [128, 108, 102], [34, 40, 0],
    [191, 86, 80], [232, 48, 0], [102, 121, 109], [218, 0, 124],
    [255, 26, 89], [138, 219, 180], [30, 2, 0], [91, 78, 81],
    [200, 149, 197], [50, 0, 51], [255, 104, 50], [102, 225, 211],
    [207, 205, 172], [208, 172, 148], [126, 211, 121], [1, 44, 88],
    [122, 123, 255], [214, 142, 1], [53, 51, 57], [120, 175, 161],
    [254, 178, 198], [117, 121, 124], [131, 115, 147], [148, 58, 77],
    [181, 244, 255], [210, 220, 213], [149, 86, 189], [106, 113, 74],
    [0, 19, 37], [2, 82, 95], [10, 163, 247], [233, 129, 118],
    [219, 213, 221], [94, 188, 209], [61, 79, 68], [126, 100, 5],
    [2, 104, 78], [150, 43, 117], [141, 133, 70], [150, 149, 197],
    [231, 115, 206], [216, 106, 120], [62, 137, 190], [202, 131, 78],
    [81, 138, 135], [91, 17, 60], [85, 129, 59], [231, 4, 196],
    [0, 0, 95], [169, 115, 153], [75, 129, 96], [89, 115, 138],
    [255, 93, 167], [247, 201, 191], [100, 49, 39], [81, 58, 1],
    [107, 148, 170], [81, 160, 88], [164, 91, 2], [29, 23, 2],
    [226, 0, 39], [231, 171, 99], [76, 96, 1], [156, 105, 102],
    [100, 84, 123], [151, 151, 158], [0, 106, 102], [57, 20, 6],
    [244, 215, 73], [0, 69, 210], [0, 108, 49], [221, 182, 208],
    [124, 101, 113], [159, 178, 164], [0, 216, 145], [21, 160, 138],
    [188, 101, 233], [255, 255, 254], [198, 220, 153], [32, 59, 60],
    [103, 17, 144], [107, 58, 100], [245, 225, 255], [255, 160, 242],
    [204, 170, 53], [55, 69, 39], [139, 180, 0], [121, 120, 104],
    [198, 0, 90], [59, 0, 10], [200, 98, 64], [41, 96, 124],
    [64, 35, 52], [125, 90, 68], [204, 184, 124], [184, 129, 131],
    [170, 81, 153], [181, 214, 195], [163, 132, 105], [159, 148, 240],
    [167, 69, 113], [184, 148, 166], [113, 187, 140], [0, 180, 51],
    [120, 158, 201], [109, 128, 186], [149, 63, 0], [94, 255, 3],
    [228, 255, 252], [27, 225, 119], [188, 177, 229], [118, 145, 47],
    [0, 49, 9], [0, 96, 205], [210, 0, 150], [137, 85, 99],
    [41, 32, 29], [91, 50, 19], [167, 111, 66], [137, 65, 46],
    [26, 58, 42], [73, 75, 90], [168, 140, 133], [244, 171, 170],
    [163, 243, 171], [0, 198, 200], [234, 139, 102], [149, 138, 159],
    [189, 201, 210], [159, 160, 100], [190, 71, 0], [101, 129, 136],
    [131, 164, 133], [69, 60, 35], [71, 103, 93], [58, 63, 0],
    [6, 18, 3], [223, 251, 113], [134, 142, 126], [152, 208, 88],
    [108, 143, 125], [215, 191, 194], [60, 62, 110], [216, 61, 102],
    [47, 93, 155], [108, 94, 70], [210, 91, 136], [91, 101, 108],
    [0, 181, 127], [84, 92, 70], [134, 96, 151], [54, 93, 37],
    [37, 47, 153], [0, 204, 255], [103, 78, 96], [252, 0, 156],
    [146, 137, 107], [30, 35, 36], [222, 201, 178], [157, 73, 72],
    [133, 171, 180], [52, 33, 66], [208, 150, 133], [164, 172, 172],
    [0, 255, 255], [174, 156, 134], [116, 42, 51], [14, 114, 197],
    [175, 216, 236], [192, 100, 185], [145, 2, 140], [254, 237, 191],
    [255, 183, 137], [156, 184, 228], [175, 255, 209], [42, 54, 76],
    [79, 74, 67], [100, 112, 149], [52, 187, 255], [128, 119, 129],
    [146, 0, 3], [179, 165, 167], [1, 134, 21], [241, 255, 200],
    [151, 111, 92], [255, 59, 193], [255, 95, 107], [7, 125, 132],
    [245, 109, 147], [87, 113, 218], [78, 30, 42], [131, 0, 85],
    [2, 211, 70], [190, 69, 45], [0, 144, 94], [190, 0, 40],
    [110, 150, 227], [0, 118, 153], [254, 201, 109], [156, 106, 125],
    [63, 161, 184], [137, 61, 227], [121, 180, 214], [127, 212, 217],
    [103, 81, 187], [178, 141, 45], [226, 122, 5], [221, 156, 184],
    [170, 188, 122], [152, 0, 52], [86, 26, 2], [143, 127, 0],
    [99, 80, 0], [205, 125, 174], [138, 94, 45], [255, 179, 225],
    [107, 100, 102], [198, 211, 0], [1, 0, 226], [136, 236, 105],
    [143, 204, 190], [33, 0, 28], [81, 31, 77], [227, 246, 227],
    [255, 142, 177], [107, 79, 41], [163, 127, 70], [106, 89, 80],
    [31, 42, 26], [4, 120, 77], [16, 24, 53], [230, 224, 208],
    [255, 116, 254], [0, 164, 95], [143, 93, 248], [75, 0, 89],
    [65, 47, 35], [216, 147, 158], [219, 157, 114], [96, 65, 67],
    [181, 186, 206], [152, 158, 183], [210, 196, 219], [165, 135, 175],
    [119, 215, 150], [127, 140, 148], [255, 155, 3], [85, 81, 150],
    [49, 221, 174], [116, 182, 113], [128, 38, 71], [42, 55, 63],
    [1, 74, 104], [105, 102, 40], [76, 123, 109], [0, 44, 39],
    [122, 69, 34], [59, 88, 89], [229, 211, 129], [255, 243, 255],
    [103, 159, 160], [38, 19, 0], [44, 87, 66], [145, 49, 175],
    [175, 93, 136], [199, 112, 106], [97, 171, 31], [140, 242, 212],
    [197, 217, 184], [159, 255, 251], [191, 69, 204], [73, 57, 65],
    [134, 59, 96], [185, 0, 118], [0, 49, 119], [197, 130, 210],
    [193, 179, 148], [96, 43, 112], [136, 120, 104], [186, 191, 176],
    [3, 0, 18], [209, 172, 254], [127, 222, 254], [75, 92, 113],
    [163, 160, 151], [230, 109, 83], [99, 123, 93], [146, 190, 165],
    [0, 248, 179], [190, 221, 255], [61, 181, 167], [221, 50, 72],
    [182, 228, 222], [66, 119, 69], [89, 140, 90], [185, 76, 89],
    [129, 129, 213], [148, 136, 139], [254, 214, 189], [83, 109, 49],
    [110, 255, 146], [228, 232, 255], [32, 226, 0], [255, 208, 242],
    [76, 131, 161], [189, 115, 34], [145, 92, 78], [140, 71, 135],
    [2, 81, 23], [162, 170, 69], [45, 27, 33], [169, 221, 176],
    [255, 79, 120], [82, 133, 0], [0, 154, 46], [23, 252, 228],
    [113, 85, 90], [82, 93, 130], [0, 25, 90], [150, 120, 116],
    [85, 85, 88], [11, 33, 44], [30, 32, 43], [239, 191, 196],
    [111, 151, 85], [111, 117, 134], [80, 29, 29], [55, 45, 0],
    [116, 29, 22], [94, 179, 147], [181, 180, 0], [221, 74, 56],
    [54, 61, 255], [173, 101, 82], [102, 53, 175], [131, 107, 186],
    [152, 170, 127], [70, 72, 54], [50, 44, 62], [124, 185, 186],
    [91, 105, 101], [112, 125, 61], [122, 0, 29], [110, 70, 54],
    [68, 58, 56], [174, 129, 255], [72, 144, 121], [137, 115, 52],
    [0, 144, 135], [218, 113, 60], [54, 22, 24], [255, 111, 1],
    [0, 102, 121], [55, 14, 119], [75, 58, 131], [201, 226, 230],
    [196, 65, 112], [255, 69, 38], [115, 190, 84], [196, 223, 114],
    [173, 255, 96], [0, 68, 125], [220, 206, 201], [189, 148, 121],
    [101, 110, 91], [236, 82, 0], [255, 110, 194], [122, 97, 126],
    [221, 174, 162], [119, 131, 127], [165, 51, 39], [96, 142, 255],
    [181, 153, 215], [165, 1, 73], [78, 0, 37], [201, 177, 169],
    [3, 145, 154], [27, 42, 37], [229, 0, 241], [152, 46, 11],
    [182, 113, 128], [224, 88, 89], [0, 96, 57], [87, 143, 155],
    [48, 82, 48], [206, 147, 76], [179, 194, 190], [192, 186, 192],
    [181, 6, 211], [23, 12, 16], [76, 83, 79], [34, 68, 81],
    [62, 65, 65], [120, 114, 109], [182, 96, 43], [32, 4, 65],
    [221, 181, 136], [73, 114, 0], [197, 170, 182], [3, 60, 97],
    [113, 178, 245], [169, 224, 136], [73, 121, 176], [162, 195, 223],
    [120, 65, 73], [45, 43, 23], [62, 14, 47], [87, 52, 76],
    [0, 145, 190], [228, 81, 209], [75, 75, 106], [92, 1, 26],
    [124, 128, 96], [255, 148, 145], [76, 50, 93], [0, 92, 139],
    [229, 253, 164], [104, 209, 182], [3, 38, 65], [20, 0, 35],
    [134, 131, 169], [207, 255, 0], [167, 44, 62], [52, 71, 90],
    [177, 187, 154], [180, 160, 79], [141, 145, 142], [161, 104, 166],
    [129, 61, 58], [66, 82, 24], [218, 131, 134], [119, 97, 51],
    [86, 57, 48], [132, 152, 174], [144, 193, 211], [181, 102, 107],
    [155, 88, 94], [133, 100, 101], [173, 124, 144], [226, 188, 0],
    [227, 170, 224], [178, 194, 254], [253, 0, 57], [0, 155, 117],
    [255, 244, 109], [232, 126, 172], [223, 227, 230], [132, 133, 144],
    [170, 146, 151], [131, 161, 147], [87, 121, 119], [62, 113, 88],
    [198, 66, 137], [234, 0, 114], [196, 168, 203], [85, 200, 153],
    [231, 143, 207], [0, 69, 71], [246, 226, 227], [150, 103, 22],
    [55, 143, 219], [67, 94, 106], [218, 0, 4], [27, 0, 15],
    [91, 156, 143], [110, 43, 82], [1, 17, 21], [227, 232, 196],
    [174, 59, 133], [234, 28, 169], [255, 158, 107], [69, 125, 139],
    [146, 103, 139], [0, 205, 187], [156, 204, 4], [0, 46, 56],
    [150, 197, 127], [207, 246, 180], [73, 40, 24], [118, 110, 82],
    [32, 55, 14], [227, 209, 159], [46, 60, 48], [178, 234, 206],
    [243, 189, 164], [162, 78, 61], [151, 111, 217], [140, 159, 168],
    [124, 43, 115], [78, 95, 55], [93, 84, 98], [144, 149, 111],
    [106, 167, 118], [219, 203, 246], [218, 113, 255], [152, 124, 149],
    [82, 50, 60], [187, 60, 66], [88, 77, 57], [79, 193, 95],
    [162, 185, 193], [121, 219, 33], [29, 89, 88], [189, 116, 78],
    [22, 11, 0], [32, 34, 26], [107, 130, 149], [0, 224, 228],
    [16, 36, 1], [27, 120, 42], [218, 169, 181], [176, 65, 93],
    [133, 146, 83], [151, 160, 148], [6, 227, 196], [71, 104, 140],
    [124, 103, 85], [7, 92, 0], [117, 96, 213], [125, 159, 0],
    [195, 109, 150], [77, 145, 62], [95, 66, 118], [252, 228, 200],
    [48, 48, 82], [79, 56, 27], [229, 165, 50], [112, 102, 144],
    [170, 154, 146], [35, 115, 99], [115, 1, 62], [255, 144, 121],
    [167, 154, 116], [2, 155, 219], [255, 1, 105], [199, 210, 231],
    [202, 136, 105], [128, 255, 205], [187, 31, 105], [144, 176, 171],
    [125, 116, 169], [252, 199, 219], [153, 55, 91], [0, 171, 77],
    [171, 174, 209], [190, 157, 145], [230, 229, 167], [51, 44, 34],
    [221, 88, 123], [245, 255, 247], [93, 48, 51], [109, 56, 0],
    [255, 0, 32], [181, 123, 179], [215, 255, 230], [197, 53, 169],
    [38, 0, 9], [106, 135, 129], [168, 171, 180], [212, 82, 98],
    [121, 75, 97], [70, 33, 178], [141, 164, 219], [199, 200, 144],
    [111, 233, 173], [162, 67, 167], [178, 176, 129], [24, 27, 0],
    [40, 97, 84], [76, 164, 59], [106, 149, 115], [168, 68, 29],
    [92, 114, 123], [115, 134, 113], [208, 207, 203], [137, 123, 119],
    [31, 63, 34], [65, 69, 167], [218, 152, 148], [161, 117, 122],
    [99, 36, 60], [173, 170, 255], [0, 205, 226], [221, 188, 98],
    [105, 142, 177], [32, 132, 98], [0, 183, 224], [97, 74, 68],
    [155, 187, 87], [122, 92, 84], [133, 122, 80], [118, 107, 126],
    [1, 72, 51], [255, 131, 71], [122, 142, 186], [39, 71, 64],
    [148, 100, 68], [235, 216, 230], [100, 98, 65], [55, 57, 23],
    [106, 212, 80], [129, 129, 123], [212, 153, 227], [151, 148, 64],
    [1, 26, 18], [82, 101, 84], [181, 136, 92], [164, 153, 165],
    [3, 173, 137], [179, 0, 139], [227, 196, 181], [150, 83, 31],
    [134, 113, 117], [116, 86, 158], [97, 125, 159], [231, 4, 82],
    [6, 126, 175], [166, 151, 182], [183, 135, 168], [156, 255, 147],
    [49, 29, 25], [58, 148, 89], [110, 116, 110], [176, 197, 174],
    [132, 237, 247], [237, 52, 136], [117, 76, 120], [56, 70, 68],
    [199, 132, 123], [0, 182, 197], [127, 166, 112], [193, 175, 158],
    [42, 127, 255], [114, 165, 140], [255, 192, 127], [157, 235, 221],
    [217, 124, 142], [126, 124, 147], [98, 230, 116], [181, 99, 158],
    [255, 168, 97], [194, 165, 128], [141, 156, 131], [183, 5, 70],
    [55, 43, 46], [0, 152, 255], [152, 89, 117], [32, 32, 76],
    [255, 108, 96], [68, 80, 131], [133, 2, 170], [114, 54, 31],
    [150, 118, 163], [72, 68, 73], [206, 214, 194], [59, 22, 74],
    [204, 167, 99], [44, 127, 119], [2, 34, 123], [163, 126, 111],
    [205, 230, 220], [205, 255, 251], [190, 129, 26], [247, 113, 131],
    [237, 230, 226], [205, 198, 180], [255, 224, 158], [58, 114, 113],
    [255, 123, 89], [78, 78, 1], [74, 198, 132], [139, 200, 145],
    [188, 138, 150], [207, 99, 83], [220, 222, 92], [94, 170, 221],
    [246, 160, 173], [226, 105, 170], [163, 218, 228], [67, 110, 131],
    [0, 46, 23], [236, 251, 255], [161, 194, 182], [80, 0, 63],
    [113, 105, 91], [103, 196, 187], [83, 110, 255], [93, 90, 72],
    [137, 0, 57], [150, 147, 129], [55, 21, 33], [94, 70, 101],
    [170, 98, 195], [141, 111, 129], [44, 97, 53], [65, 6, 1],
    [86, 70, 32], [230, 144, 52], [109, 166, 189], [229, 142, 86],
    [227, 166, 139], [72, 177, 118], [210, 125, 103], [181, 178, 104],
    [127, 132, 39], [255, 132, 230], [67, 87, 64], [234, 228, 8],
    [244, 245, 255], [50, 88, 0], [75, 107, 165], [173, 206, 255],
    [155, 138, 204], [136, 81, 56], [88, 117, 193], [126, 115, 17],
    [254, 165, 202], [159, 139, 91], [165, 91, 84], [137, 0, 106],
    [175, 117, 111], [42, 32, 0], [116, 153, 161], [255, 181, 80],
    [0, 1, 30], [209, 81, 28], [104, 129, 81], [188, 144, 138],
    [120, 200, 235], [133, 2, 255], [72, 61, 48], [196, 34, 33],
    [94, 167, 255], [120, 87, 21], [12, 234, 145], [255, 250, 237],
    [179, 175, 157], [62, 61, 82], [90, 155, 194], [156, 47, 144],
    [141, 87, 0], [173, 215, 156], [0, 118, 139], [51, 125, 0],
    [197, 151, 0], [49, 86, 220], [148, 69, 117], [236, 255, 220],
    [210, 76, 178], [151, 112, 60], [76, 37, 127], [158, 3, 102],
    [136, 255, 236], [181, 100, 129], [57, 109, 43], [86, 115, 95],
    [152, 131, 118], [155, 177, 149], [169, 121, 92], [228, 197, 211],
    [159, 79, 103], [30, 43, 57], [102, 67, 39], [175, 206, 120],
    [50, 46, 223], [134, 180, 135], [194, 48, 0], [171, 232, 107],
    [150, 101, 109], [37, 14, 53], [166, 0, 25], [0, 128, 207],
    [202, 239, 255], [50, 63, 97], [164, 73, 220], [106, 157, 59],
    [255, 90, 228], [99, 106, 1], [209, 108, 218], [115, 96, 96],
    [255, 186, 173], [211, 105, 180], [255, 222, 214], [108, 109, 116],
    [146, 125, 94], [132, 93, 112], [91, 98, 193], [47, 74, 54],
    [228, 95, 53], [255, 59, 83], [172, 132, 221], [118, 41, 136],
    [112, 236, 152], [64, 133, 67], [44, 53, 51], [46, 24, 45],
    [50, 57, 37], [25, 24, 27], [47, 46, 44], [2, 60, 50],
    [155, 158, 226], [88, 175, 173], [92, 66, 77], [122, 197, 166],
    [104, 93, 117], [185, 188, 189], [131, 67, 87], [26, 123, 66],
    [46, 87, 170], [229, 81, 153], [49, 110, 71], [205, 0, 197],
    [106, 0, 77], [127, 187, 236], [243, 86, 145], [215, 197, 74],
    [98, 172, 183], [203, 161, 188], [162, 138, 154], [108, 63, 59],
    [255, 228, 125], [220, 186, 227], [95, 129, 109], [58, 64, 74],
    [125, 191, 50], [230, 236, 220], [133, 44, 25], [40, 83, 102],
    [184, 203, 156], [14, 13, 0], [75, 93, 86], [107, 84, 63],
    [226, 113, 114], [5, 104, 236], [46, 181, 0], [210, 22, 86],
    [239, 175, 255], [104, 32, 33], [45, 32, 17], [218, 76, 255],
    [112, 150, 142], [255, 123, 125], [74, 25, 48], [232, 194, 130],
    [231, 219, 188], [166, 132, 134], [31, 38, 60], [54, 87, 78],
    [82, 206, 121], [173, 170, 169], [138, 159, 69], [101, 66, 210],
    [0, 251, 140], [93, 105, 123], [204, 210, 127], [148, 165, 161],
    [121, 2, 41], [227, 131, 230], [126, 164, 193], [78, 68, 82],
    [75, 44, 0], [98, 11, 112], [49, 76, 30], [135, 74, 166],
    [227, 0, 145], [102, 70, 10], [235, 154, 139], [234, 195, 163],
    [152, 234, 179], [171, 145, 128], [184, 85, 47], [26, 43, 47],
    [148, 221, 197], [157, 140, 118], [156, 131, 51], [148, 169, 201],
    [57, 41, 53], [140, 103, 94], [204, 233, 58], [145, 113, 0],
    [1, 64, 11], [68, 152, 150], [28, 163, 112], [224, 141, 167],
    [139, 74, 78], [102, 119, 118], [70, 146, 173], [103, 189, 168],
    [105, 37, 92], [211, 191, 255], [74, 81, 50], [126, 146, 133],
    [119, 115, 60], [231, 160, 204], [81, 162, 136], [44, 101, 106],
    [77, 92, 94], [201, 64, 58], [221, 215, 243], [0, 88, 68],
    [180, 162, 0], [72, 143, 105], [133, 129, 130], [212, 233, 185],
    [61, 115, 151], [202, 232, 206], [214, 0, 52], [170, 103, 70],
    [158, 85, 133], [186, 98, 0]
]

high_contrast_arr = np.array(high_contrast, dtype=np.uint8)
