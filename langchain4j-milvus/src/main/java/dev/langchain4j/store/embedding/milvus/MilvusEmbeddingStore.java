package dev.langchain4j.store.embedding.milvus;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static dev.langchain4j.store.embedding.milvus.CollectionOperationsExecutor.*;
import static dev.langchain4j.store.embedding.milvus.CollectionRequestBuilder.buildSearchRequest;
import static dev.langchain4j.store.embedding.milvus.Generator.generateRandomIds;
import static dev.langchain4j.store.embedding.milvus.Mapper.*;
import static io.milvus.common.clientenum.ConsistencyLevelEnum.EVENTUALLY;
import static io.milvus.param.IndexType.FLAT;
import static io.milvus.param.MetricType.COSINE;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.internal.Utils;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.filter.Filter;
import io.milvus.client.MilvusServiceClient;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import org.apache.commons.lang3.StringUtils;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents an <a href="https://milvus.io/">Milvus</a> index as an embedding store.
 * <br>
 * Supports both local and <a href="https://zilliz.com/">managed</a> Milvus instances.
 * <br>
 * Supports storing {@link Metadata} and filtering by it using a {@link Filter}
 * (provided inside an {@link EmbeddingSearchRequest}).
 */
public class MilvusEmbeddingStore implements EmbeddingStore<TextSegment> {

  static final String ID_FIELD_NAME = "id";
  static final String TEXT_FIELD_NAME = "text";
  static final String METADATA_FIELD_NAME = "metadata";
  static final String VECTOR_FIELD_NAME = "vector";

  private final MilvusServiceClient milvusClient;
  private final String collectionName;
  private final MetricType metricType;
  private final ConsistencyLevelEnum consistencyLevel;
  private final boolean retrieveEmbeddingsOnSearch;

  public MilvusEmbeddingStore(
    String host,
    Integer port,
    String collectionName,
    Integer dimension,
    IndexType indexType,
    MetricType metricType,
    String uri,
    String token,
    String username,
    String password,
    ConsistencyLevelEnum consistencyLevel,
    Boolean retrieveEmbeddingsOnSearch,
    String databaseName
  ) {
    ConnectParam.Builder connectBuilder = ConnectParam
      .newBuilder()
      .withHost(getOrDefault(host, "localhost"))
      .withPort(getOrDefault(port, 19530))
      .withUri(uri)
      .withToken(token)
      .withAuthorization(username, password);

    if (databaseName != null) {
      connectBuilder.withDatabaseName(databaseName);
    }

    this.milvusClient = new MilvusServiceClient(connectBuilder.build());
    this.collectionName = getOrDefault(collectionName, "default");
    this.metricType = getOrDefault(metricType, COSINE);
    this.consistencyLevel = getOrDefault(consistencyLevel, EVENTUALLY);
    this.retrieveEmbeddingsOnSearch = getOrDefault(retrieveEmbeddingsOnSearch, false);

    if (!hasCollection(milvusClient, this.collectionName)) {
      createCollection(milvusClient, this.collectionName, ensureNotNull(dimension, "dimension"));
      createIndex(milvusClient, this.collectionName, getOrDefault(indexType, FLAT), this.metricType);
    }

    loadCollectionInMemory(milvusClient, collectionName);
  }

  public void dropCollection(String collectionName) {
    CollectionOperationsExecutor.dropCollection(milvusClient, collectionName);
  }

  public void createPartition(String partitionName) {
    if (StringUtils.isNotEmpty(partitionName) && !hasPartition(this.milvusClient, this.collectionName, partitionName)) {
      CollectionOperationsExecutor.createPartition(this.milvusClient, this.collectionName, partitionName);
    }
  }

  public String add(Embedding embedding) {
    String id = Utils.randomUUID();
    add(id, embedding);
    return id;
  }

  public void add(String id, Embedding embedding) {
    addInternal(id, embedding, null);
  }

  public String add(Embedding embedding, TextSegment textSegment) {
    String id = Utils.randomUUID();
    addInternal(id, embedding, textSegment);
    return id;
  }

  public List<String> addAll(List<Embedding> embeddings) {
    List<String> ids = generateRandomIds(embeddings.size());
    addAllInternal(ids, embeddings, null, null);
    return ids;
  }

  public List<String> addAll(List<Embedding> embeddings, List<TextSegment> embedded) {
    List<String> ids = generateRandomIds(embeddings.size());
    addAllInternal(ids, embeddings, embedded, null);
    return ids;
  }

public String add(Embedding embedding, String partitionName) {
    String id = Utils.randomUUID();
    add(id, embedding, partitionName);
    return id;
  }

  public void add(String id, Embedding embedding, String partitionName) {
    addInternal(id, embedding, null, partitionName);
  }

  public String add(Embedding embedding, TextSegment textSegment, String partitionName) {
    String id = Utils.randomUUID();
    addInternal(id, embedding, textSegment, partitionName);
    return id;
  }

  public List<String> addAll(List<Embedding> embeddings, String partitionName) {
    List<String> ids = generateRandomIds(embeddings.size());
    addAllInternal(ids, embeddings, null, partitionName);
    return ids;
  }

  public List<String> addAll(List<Embedding> embeddings, List<TextSegment> embedded, String partitionName) {
    List<String> ids = generateRandomIds(embeddings.size());
    addAllInternal(ids, embeddings, embedded, partitionName);
    return ids;
  }

  @Override
  public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest embeddingSearchRequest) {
    return this.search(embeddingSearchRequest, null);
  }

  public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest embeddingSearchRequest, List<String> partitionNames) {

    SearchParam searchParam = buildSearchRequest(
            collectionName,
            partitionNames,
            embeddingSearchRequest.queryEmbedding().vectorAsList(),
            embeddingSearchRequest.filter(),
            embeddingSearchRequest.maxResults(),
            metricType,
            consistencyLevel
    );

    SearchResultsWrapper resultsWrapper = CollectionOperationsExecutor.search(milvusClient, searchParam);

    List<EmbeddingMatch<TextSegment>> matches = toEmbeddingMatches(
            milvusClient,
            resultsWrapper,
            collectionName,
            consistencyLevel,
            retrieveEmbeddingsOnSearch
    );

    List<EmbeddingMatch<TextSegment>> result = matches.stream()
            .filter(match -> match.score() >= embeddingSearchRequest.minScore())
            .collect(toList());

    return new EmbeddingSearchResult<>(result);
  }

  private void addInternal(String id, Embedding embedding, TextSegment textSegment) {
    addAllInternal(
      singletonList(id),
      singletonList(embedding),
      textSegment == null ? null : singletonList(textSegment),
      null
    );
  }

    private void addInternal(String id, Embedding embedding, TextSegment textSegment, String partitionName) {
    addAllInternal(
      singletonList(id),
      singletonList(embedding),
      textSegment == null ? null : singletonList(textSegment),
      partitionName
    );
  }

  private void addAllInternal(List<String> ids, List<Embedding> embeddings, List<TextSegment> textSegments, String partitionName) {
    List<InsertParam.Field> fields = new ArrayList<>();
    fields.add(new InsertParam.Field(ID_FIELD_NAME, ids));
    fields.add(new InsertParam.Field(TEXT_FIELD_NAME, toScalars(textSegments, ids.size())));
    fields.add(new InsertParam.Field(METADATA_FIELD_NAME, toMetadataJsons(textSegments, ids.size())));
    fields.add(new InsertParam.Field(VECTOR_FIELD_NAME, toVectors(embeddings)));

    insert(milvusClient, collectionName, fields, partitionName);
    flush(milvusClient, collectionName);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {

    private String host;
    private Integer port;
    private String collectionName;
    private Integer dimension;
    private IndexType indexType;
    private MetricType metricType;
    private String uri;
    private String token;
    private String username;
    private String password;
    private ConsistencyLevelEnum consistencyLevel;
    private Boolean retrieveEmbeddingsOnSearch;
    private String databaseName;

    /**
     * @param host The host of the self-managed Milvus instance.
     *             Default value: "localhost".
     * @return builder
     */
    public Builder host(String host) {
      this.host = host;
      return this;
    }

    /**
     * @param port The port of the self-managed Milvus instance.
     *             Default value: 19530.
     * @return builder
     */
    public Builder port(Integer port) {
      this.port = port;
      return this;
    }

    /**
     * @param collectionName The name of the Milvus collection.
     *                       If there is no such collection yet, it will be created automatically.
     *                       Default value: "default".
     * @return builder
     */
    public Builder collectionName(String collectionName) {
      this.collectionName = collectionName;
      return this;
    }

    /**
     * @param dimension The dimension of the embedding vector. (e.g. 384)
     *                  Mandatory if a new collection should be created.
     * @return builder
     */
    public Builder dimension(Integer dimension) {
      this.dimension = dimension;
      return this;
    }

    /**
     * @param indexType The type of the index.
     *                  Default value: FLAT.
     * @return builder
     */
    public Builder indexType(IndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    /**
     * @param metricType The type of the metric used for similarity search.
     *                   Default value: COSINE.
     * @return builder
     */
    public Builder metricType(MetricType metricType) {
      this.metricType = metricType;
      return this;
    }

    /**
     * @param uri The URI of the managed Milvus instance. (e.g. "https://xxx.api.gcp-us-west1.zillizcloud.com")
     * @return builder
     */
    public Builder uri(String uri) {
      this.uri = uri;
      return this;
    }

    /**
     * @param token The token (API key) of the managed Milvus instance.
     * @return builder
     */
    public Builder token(String token) {
      this.token = token;
      return this;
    }

    /**
     * @param username The username. See details <a href="https://milvus.io/docs/authenticate.md">here</a>.
     * @return builder
     */
    public Builder username(String username) {
      this.username = username;
      return this;
    }

    /**
     * @param password The password. See details <a href="https://milvus.io/docs/authenticate.md">here</a>.
     * @return builder
     */
    public Builder password(String password) {
      this.password = password;
      return this;
    }

    /**
     * @param consistencyLevel The consistency level used by Milvus.
     *                         Default value: EVENTUALLY.
     * @return builder
     */
    public Builder consistencyLevel(ConsistencyLevelEnum consistencyLevel) {
      this.consistencyLevel = consistencyLevel;
      return this;
    }

    /**
     * @param retrieveEmbeddingsOnSearch During a similarity search in Milvus (when calling findRelevant()),
     *                                   the embedding itself is not retrieved.
     *                                   To retrieve the embedding, an additional query is required.
     *                                   Setting this parameter to "true" will ensure that embedding is retrieved.
     *                                   Be aware that this will impact the performance of the search.
     *                                   Default value: false.
     * @return builder
     */
    public Builder retrieveEmbeddingsOnSearch(Boolean retrieveEmbeddingsOnSearch) {
      this.retrieveEmbeddingsOnSearch = retrieveEmbeddingsOnSearch;
      return this;
    }

    /**
     * @param databaseName Milvus name of database.
     *                     Default value: null. In this case default Milvus database name will be used.
     * @return builder
     */
    public Builder databaseName(String databaseName) {
      this.databaseName = databaseName;
      return this;
    }

    public MilvusEmbeddingStore build() {
      return new MilvusEmbeddingStore(
        host,
        port,
        collectionName,
        dimension,
        indexType,
        metricType,
        uri,
        token,
        username,
        password,
        consistencyLevel,
        retrieveEmbeddingsOnSearch,
        databaseName
      );
    }
  }
}
